from datetime import datetime
import torch
from pathlib import Path
import gc
from dataset import IMDBBertDataset
from cdataset import classificationDataset
from Bert import BERT
from cBert import BERTForclassification
from cBertTrainer import BertTrainerclassification
import sys

EMB_SIZE = 512
HIDDEN_SIZE = 256
EPOCHS = 4
BATCH_SIZE = 12
NUM_HEADS = 16
data = sys.argv[1]
hex = sys.argv[2]
datatype = sys.argv[3]
sdata = sys.argv[4]
numusers = sys.argv[5]
#CHECKPOINT_DIR = '/content/drive/MyDrive/Bertclass_data '

timestamp = datetime.utcnow().timestamp()
#LOG_DIR = 'data/logs/bert'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gc.collect()



if __name__ == '__main__':
    print("Prepare dataset")
    gc.collect()
    #torch.cuda.empty_cache()
    ds = ds = IMDBBertDataset("dataset/ST-BERT "+datatype+"/" +hex+"/"+ data +f"train+test_{numusers}_users" +".csv", spath=sdata)
    ds_full = classificationDataset("dataset/ST-BERT "+datatype+"/" +hex+"/"+ data +f"train+test_{numusers}_users" +".csv", word_vocab= ds.vocab,spath=sdata)

    ds_test = classificationDataset("dataset/ST-BERT "+datatype+"/" +hex+"/"+ data +f"test_{numusers}_users" +".csv",word_vocab= ds.vocab,user_dataset = ds_full.user_vocab, poi_vocab = ds_full.vocab_poi,spath=sdata)
    ds_train = classificationDataset("dataset/ST-BERT "+datatype+"/" +hex+"/"+ data +f"train_{numusers}_users" +".csv",word_vocab= ds.vocab,user_dataset = ds_full.user_vocab, poi_vocab = ds_full.vocab_poi,spath=sdata)
    print(ds_full.poi_size)
    print("==============")
    print(len(ds_full.user_vocab))
    #print(torch.cuda.memory_allocated(0)/1024/1024/1024)
    bert = BERT(len(ds.vocab), EMB_SIZE, HIDDEN_SIZE, NUM_HEADS, poi_input=ds_full.poi_size,spath=sdata).to(device)
    #checkpoint = torch.load("/content/drive/MyDrive/dataset/Bert_checkPoints/Bert_v1.11_TKY_hex9_512_trained_on_explict-512-spatial-only")
    #bert.load_state_dict(checkpoint['model_state_dict'])#,strict=False
    bertclas=BERTForclassification(bert,len(ds_full.user_vocab),EMB_SIZE, HIDDEN_SIZE).to(device)
    #model_temp = RotoGrad(bert.get_submodule("sub"), [bert.get_submodule("token_prediction_layer"), bert.get_submodule("spatial_prediction_layer")],64*2, normalize_losses=True).to(device)
    trainer = BertTrainerclassification(
        model=bertclas,
        dataset=ds_train,
        log_dir="None",
        checkpoint_dir="None",
        print_progress_every=100,
        print_accuracy_every=100,
        batch_size=BATCH_SIZE,
        learning_rate=0.00002,
        epochs=40,
        testdataset=ds_test

    )
    trainer.print_summary()
    trainer()