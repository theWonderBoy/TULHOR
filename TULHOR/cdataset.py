import random
import typing
from collections import Counter
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import torch
import pickle
from tqdm import tqdm
from torch.utils.data import Dataset
from torchtext.vocab import vocab
from torchtext.data.utils import get_tokenizer
from gensim.models import Word2Vec
import h3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class classificationDataset(Dataset):
    CLS = '[CLS]'
    PAD = '[PAD]'
    SEP = '[SEP]'
    MASK = '[MASK]'
    UNK = '[UNK]'

    MASK_PERCENTAGE = 0.4

    MASKED_INDICES_COLUMN = 'masked_indices'
    TARGET_COLUMN = 'indices'
    NSP_TARGET_COLUMN = 'is_next'
    TOKEN_MASK_COLUMN = 'token_mask'
    USER_ID_COLUMN ='User_id'
    WEEK_DAY_COLUMN = 'WeekDay'
    TIME_COLUMN="Check_in_time"
    IMPLICIT_COLUMN= "Implicit_column"
    OPTIMAL_LENGTH_PERCENTILE = 100
    OPTIMAL_LENGTH_PERCENTILE1 = 70


    def __init__(self, path, ds_from=None, ds_to=None, should_include_text=False,word_vocab = None,user_dataset = None , poi_vocab=None , spath=None):

        self.ds: pd.Series = pd.read_csv(path,index_col=0)
        
        #print("Number of rows = {0}".format(len(self.ds)))
        #print("Number of rows = {0}".format(len(self.ds)))
        self.tokenizer = get_tokenizer(None)
        self.counter = Counter()
        self.counter1 = Counter()
        self.vocab = None
        self.poi_size = 0
        self.optimal_sentence_length = None
        self.should_include_text = should_include_text
        self.vocab_poi = poi_vocab
        self.spath =spath
        if should_include_text:
            self.columns = ['original_sentence', self.MASKED_INDICES_COLUMN, self.USER_ID_COLUMN,self.WEEK_DAY_COLUMN,self.TIME_COLUMN,self.IMPLICIT_COLUMN,"poi"]
        else:
            self.columns = [self.MASKED_INDICES_COLUMN, self.USER_ID_COLUMN,self.WEEK_DAY_COLUMN,self.TIME_COLUMN,self.IMPLICIT_COLUMN,"poi"]
        self.user_dataset = user_dataset
        self.word_vocab = word_vocab
        self.df = self.prepare_dataset()
        #print(self.df.head(5))



    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        item = self.df.iloc[idx]

        inp = torch.Tensor(item[self.MASKED_INDICES_COLUMN]).long()        
        attention_mask = (inp == self.vocab[self.PAD]).unsqueeze(0)
        poi = torch.Tensor(item["poi"]).long()
        user_item=torch.Tensor([item[self.USER_ID_COLUMN]]).long()
        weekday =torch.Tensor([item[self.WEEK_DAY_COLUMN]])
        time =torch.Tensor(item[self.TIME_COLUMN]).long()  
        exp = torch.Tensor(item[self.IMPLICIT_COLUMN]).long()
        attention_mask1 = (exp == self.vocab[self.PAD]).unsqueeze(0)
        return (
            inp.to(device),
            attention_mask.to(device),
            user_item.to(device),
            weekday.to(device),
            time.to(device),
            exp.to(device),
            attention_mask1.to(device),
            poi.to(device)
        )

    def prepare_dataset(self) -> pd.DataFrame:
        sentences = []
        sentences1 = []
        nsp = []
        sentence_lens = []
        sentences1_lens = []
        user_id = []
        week_day =[]
        time = []
        pois = []
        # Split dataset on sentences

        for index, row in self.ds.iterrows():
            self.counter1.update([str(row['user'])])
            sentences.append(row['h3'])
            sentences1.append(row['id'])
            user_id.append(str(row['user']))
            time.append(row["time"])
            week_day.append(row["weekday"])
            sentence_lens.append(len(row['h3'].split()))
            sentences1_lens.append(len(row['id'].split()))
            pois.append(row["poi"])
        self.optimal_sentence_length = self._find_optimal_sentence_length(sentence_lens, self.OPTIMAL_LENGTH_PERCENTILE)
        self.optimal_sentence_length1 = self._find_optimal_sentence_length(sentences1_lens , self.OPTIMAL_LENGTH_PERCENTILE1)
        print(f"Optimal length exp = {self.optimal_sentence_length}")
        print(f"Optimal length imp = {self.optimal_sentence_length1}")
        print("Create vocabulary")

        for sentence in tqdm(pois):
            s = sentence.split()
            self.counter.update(s)
        self._fill_poi_vocab()
        self._fill_vocab_pre_trained("Spatial-Extractor/" + self.spath)
        if self.user_dataset == None:
          self._fill_user_vocab()
        else:
          self.user_vocab= self.user_dataset
        

        print("Preprocessing dataset")
        for i in tqdm(range(0,len(sentences))):
            sentence = self.tokenizer(sentences[i])
            sentence_imp = self.tokenizer(sentences1[i])
            nsp.append(self._create_item(sentence, user_id[i]) + (week_day[i],) + self.process_time(time[i]) + (self._create_item1(sentence_imp, user_id[i]),) + (self._create_item_POI(pois[i].split()),))

        df = pd.DataFrame(nsp, columns=self.columns)
        return df

    def _update_length(self, sentences: typing.List[str], lengths: typing.List[int]):
        lengths.append(len(sentences))
        return lengths
    def process_time(self,time_seq):
        time_seq = time_seq.split()
        #print(time_seq)
        result = [0 for i in range(0,self.optimal_sentence_length)]
        #print(len(result))
        #print(len(time_seq))
        for i in range(0,len(time_seq)):
          #print(datetime.strptime(time_seq[i],'%H:%M:%S').minute)
          result[i] =  datetime.strptime(time_seq[i],'%H:%M:%S').minute*60 +datetime.strptime(time_seq[i],'%H:%M:%S').hour*60*60+datetime.strptime(time_seq[i],'%H:%M:%S').second
        return (result,)


    def _find_optimal_sentence_length(self, lengths: typing.List[int], perce):
        arr = np.array(lengths)
        return int(np.percentile(arr, perce))
    def _fill_user_vocab(self):
      self.user_vocab = vocab(self.counter1, min_freq=1)
    def _fill_poi_vocab(self):
        # specials= argument is only in 0.12.0 version
        # specials=[self.CLS, self.PAD, self.MASK, self.SEP, self.UNK]
        if self.vocab_poi != None:
          return
        self.vocab_poi = vocab(self.counter, min_freq=1)
        self.vocab_poi.insert_token(self.CLS, 0)
        self.vocab_poi.insert_token(self.PAD, 1)
        self.vocab_poi.insert_token(self.MASK, 2)
        self.vocab_poi.insert_token(self.UNK, 3)
        self.vocab_poi.set_default_index(4)
        self.poi_size = len(self.vocab_poi)


    def _fill_vocab_pre_trained(self , word2vec_model_path):
        # specials= argument is only in 0.12.0 version
        # specials=[self.CLS, self.PAD, self.MASK, self.SEP, self.UNK]
        self.vocab = vocab({}, min_freq=1)
        model = Word2Vec.load(word2vec_model_path)
        for i in model.wv.index2entity:
          self.vocab.insert_token(i, len(self.vocab))
        #print(self.vocab.get_stoi())
        # 0.11.0 uses this approach to insert specials
        self.vocab.insert_token(self.CLS, len(self.vocab))
        self.vocab.insert_token(self.PAD, len(self.vocab))
        self.vocab.insert_token(self.MASK, len(self.vocab))
        self.vocab.insert_token(self.SEP, len(self.vocab))
        self.vocab.insert_token(self.UNK, len(self.vocab))
        self.vocab.set_default_index(len(self.vocab)-1)
        #print(self.vocab.get_stoi())
    def _create_item(self, sentence_inp: typing.List[str], user_id:int):
        # Create masked sentence item
        updated_sentence_inp = self._preprocess_sentence(sentence_inp.copy())
        #print(len(updated_sentence_inp))
        #updated_second, second_mask = self._preprocess_sentence(second.copy())
        token_indices = self.vocab.lookup_indices(updated_sentence_inp)
        user_id = self.user_vocab.lookup_indices([user_id])[0]


        if self.should_include_text:
            return (updated_sentence_inp,token_indices,user_id)
        else:
            return (token_indices,user_id)
    def _create_item_POI(self, poi_inp):
        updated_sentence_inp = self._preprocess_sentence(poi_inp.copy())
        #print(len(updated_sentence_inp))
        #updated_second, second_mask = self._preprocess_sentence(second.copy())
        token_indices = self.vocab_poi.lookup_indices(updated_sentence_inp)
        return token_indices

    def _preprocess_sentence(self, sentence, should_mask: bool = True):
        sentence = self._pad_sentence([self.CLS] + sentence)
        return sentence


    def _pad_sentence(self, sentence):
        len_s = len(sentence)

        if len_s >= self.optimal_sentence_length:
            s = sentence[:self.optimal_sentence_length]
        else:
            s = sentence + [self.PAD] * (self.optimal_sentence_length - len_s)

        return s
    def _create_item1(self, sentence_inp: typing.List[str], user_id:int):
        # Create masked sentence item
        updated_sentence_inp = self._preprocess_sentence1(sentence_inp.copy())
        #print(len(updated_sentence_inp))
        #updated_second, second_mask = self._preprocess_sentence(second.copy())
        token_indices = self.vocab.lookup_indices(updated_sentence_inp)
        user_id = self.user_vocab.lookup_indices([user_id])[0]


        return token_indices


    def _preprocess_sentence1(self, sentence, should_mask: bool = True):
        sentence = self._pad_sentence([self.CLS] + sentence)
        return sentence


    def _pad_sentence1(self, sentence):
        len_s = len(sentence)

        if len_s >= self.optimal_sentence_length1:
            s = sentence[:self.optimal_sentence_length1]
        else:
            s = sentence + [self.PAD] * (self.optimal_sentence_length1 - len_s)

        return s
    
    def class_weight(self):
        result = []
        for i in self:
          result.append(i[2].item())
        return [ x[1] for x in sorted(Counter(result).items()) ]  