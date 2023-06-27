import random
import typing
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import pickle
from tqdm import tqdm
from torch.utils.data import Dataset
from torchtext.vocab import vocab
from torchtext.data.utils import get_tokenizer
from gensim.models import Word2Vec
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class TULDataset(Dataset):
    TOKENIZED_SEQ = 'Tokenized_seq'
    ORIGINAL_SEQ = 'Original_seq'
    USER_LABEL = 'User'
    ORIGINAL_LENGTH = 'Original_seq_length'

    def __init__(self, path, word_dim=256):

        self.ds: pd.Series = pd.read_csv(path) 
        self.tokenizer = get_tokenizer(None)
        self.counter = Counter()
        self.vocab = None
        self.max_length = 0
        self.num_users = 0
        self.columns = [self.ORIGINAL_SEQ, self.TOKENIZED_SEQ ,self.USER_LABEL , self.ORIGINAL_LENGTH]
        self.pad_value = 0
        self.word_dim = word_dim
        self.df = self.prepare_dataset()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        item = self.df.iloc[idx]
        inp = torch.Tensor(item[self.TOKENIZED_SEQ]).long()
        user_label = torch.Tensor([item[self.USER_LABEL]]).long()
        un_padded_length = torch.Tensor([item[self.ORIGINAL_LENGTH]]).long()
        return (
            inp.to(device),
            user_label.to(device),
            un_padded_length.to(device)
            )

    def prepare_dataset(self) -> pd.DataFrame:
        output=[]
        sentences=[]
        length=[]
        users=set()
        for i in self.ds['poi']:
          sentences.append(i.split())
          length.append(len(i.split()))
        for i in self.ds['user']:
          users.add(i)
        self.num_users = len(users)
        self.max_length = max(length)
        print("Create vocabulary From word2vec model")
        model = Word2Vec(sentences=sentences, window=10, min_count=1, workers=4, size=self.word_dim)
        self._fill_vocab_pre_trained(model)
        print("Done !!")
        print("Preprocessing dataset")
        for index, row in tqdm(self.ds.iterrows(), total=self.ds.shape[0]):
            seq = self.tokenizer(row['poi'])
            output.append(self._create_item(seq,row["user"]))
        #print(output)
        df = pd.DataFrame(output, columns=self.columns)
        return df

    def _fill_vocab_pre_trained(self , model):
        self.vocab = vocab({}, min_freq=1)
        for i in model.wv.index2entity:
          self.vocab.insert_token(i, len(self.vocab))
        self.pad_value = len(self.vocab)
    
    def _create_item(self, seq: typing.List[str], user_label:int):
        # We tokenize the input, converting seq into an array of Integers
        original_seq = seq
        original_len = len(seq)
        tokenized_seq = self.vocab.lookup_indices(seq.copy())
        s = tokenized_seq + [self.pad_value] * (self.max_length - original_len)
        return (original_seq,s,user_label,original_len)