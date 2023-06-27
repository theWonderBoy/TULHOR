import time
from datetime import datetime
from pathlib import Path

import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from balanced_loss import Loss
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from cBert import BERTForclassification
from cdataset import classificationDataset
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def percentage(batch_size: int, max_index: int, current_index: int):
    """Calculate epoch progress percentage
    Args:
        batch_size: batch size
        max_index: max index in epoch
        current_index: current index
    Returns:
        Passed percentage of dataset
    """
    batched_max = max_index // batch_size
    return round(current_index / batched_max * 100, 2)


def nsp_accuracy(result: torch.Tensor, target: torch.Tensor):
    """Calculate NSP accuracy between two tensors
    Args:
        result: result calculated by model
        target: real target
    Returns:
        NSP accuracy
    """
    s = (result.argmax(1) == target.argmax(1)).sum()
    return round(float(s / result.size(0)), 2)


def token_accuracy(result: torch.Tensor, target: torch.Tensor, inverse_token_mask: torch.Tensor):
    """Calculate MLM accuracy between ONLY masked words
    Args:
        result: result calculated by model
        target: real target
        inverse_token_mask: well-known inverse token mask
    Returns:
        MLM accuracy
    """
    r = result.argmax(-1).masked_select(~inverse_token_mask)
    print(r.size())
    #print(result.size())
    t = target.masked_select(~inverse_token_mask)
    s = (r == t).sum()
    print("SUM = {0}".format(s))
    print("result.size(0) = {0}".format(result.size(0) ))
    print("result.size(1) = {0}".format(result.size(1)))
    return float(s / r.size(0))


class BertTrainerclassification:

    def __init__(self,
                 model: BERTForclassification,
                 dataset: classificationDataset,
                 log_dir: Path,
                 checkpoint_dir: Path = None,
                 print_progress_every: int = 10,
                 print_accuracy_every: int = 50,
                 batch_size: int = 24,
                 learning_rate: float = 0.005,
                 epochs: int = 5,
                 testdataset = None
                 ):
        self.model = model
        self.dataset = dataset
        self.ds_test = testdataset
        self.batch_size = batch_size
        self.epochs = epochs
        self.current_epoch = 0

        self.loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)
        self.test_loader = DataLoader(self.ds_test, batch_size=1 , shuffle=True)

        #self.writer = SummaryWriter(str(log_dir))
        self.checkpoint_dir = checkpoint_dir
        self.ml_criterion = Loss(loss_type="cross_entropy",samples_per_class=self.dataset.class_weight(),class_balanced=True)
        #self.ml_criterion = nn.CrossEntropyLoss().to(device)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        #self.optimizer1 = torch.optim.Adam(multimodel.parameters(), lr=learning_rate)

        self._splitter_size = 35

        self._ds_len = len(self.dataset)
        self._batched_len = self._ds_len // self.batch_size

        self._print_every = print_progress_every
        self._accuracy_every = print_accuracy_every

    def print_summary(self):
        ds_len = len(self.dataset)

        print("Model Summary\n")
        print('=' * self._splitter_size)
        print(f"Device: {device}")
        print(f"Training dataset len: {ds_len}")
        print(f"Max / Optimal sentence len: {self.dataset.optimal_sentence_length}")
        print(f"Vocab size: {len(self.dataset.vocab)}")
        print(f"Batch size: {self.batch_size}")
        print(f"Batched dataset len: {self._batched_len}")
        print('=' * self._splitter_size)
        print()

    def __call__(self):
        for self.current_epoch in range(self.current_epoch, self.epochs):
            #print(torch.cuda.memory_allocated(0)/1024/1024/1024)
            loss = self.train(self.current_epoch)
            
            #self.save_checkpoint(self.current_epoch, step=-1, loss=loss)
            #print(torch.cuda.memory_allocated(device))
            self.calculate_test_accuracy()

    def calculate_test_accuracy(self):
        self.model.eval()
        top1 = 0
        top3 = 0
        top5 = 0
        top7 = 0
        top10 = 0
        targets = []
        results = []
        with torch.no_grad():
          for i , value in enumerate(self.test_loader):
            input,mask,target,weekday,time_vec,imp,imp_mask,poi = value
            targets.append(target.item())
          #print(input.size())
          #print(mask.size())
          #print(target.size())
            #print(input)
           # print(mask)
           # print(poi)
            result = self.model(input,mask,time_vec,poi)
            top1 += self.find_k_accuracy(1,result,target)
            results.append(torch.topk(result, 1).indices.item())
            top3 += self.find_k_accuracy(3,result,target)
            top5 += self.find_k_accuracy(5,result,target)
            top7 += self.find_k_accuracy(7,result,target)
            top10 += self.find_k_accuracy(10,result,target)
            #print(targets)
            #print(results)
        top1 = top1/len(self.ds_test)
        top3 = top3/len(self.ds_test)
        top5 = top5/len(self.ds_test)
        top7 = top7/len(self.ds_test)
        top10 = top10/len(self.ds_test)
        f1_score_macro = f1_score(targets, results, average='macro')
        r_score = recall_score(targets, results, average='macro')
        p_score = precision_score(targets, results, average='macro')
        print(f"top@1 = {top1} | top@3 = {top3} | top@5 = {top5} | macro-P={p_score} | macro-R{r_score} |f1_score_macro={f1_score_macro}" )
        self.model.train()
        

    
    def find_k_accuracy(self,k,result,target):
      #print(result)
      #print(target)
      if target in torch.topk(result, k).indices:
        return 1
      return 0

    def train(self, epoch: int):
        print(f"Begin epoch {epoch}")
        prev = time.time()
        average_class_loss = 0
        #print(torch.cuda.memory_allocated(0)/1024/1024/1024)
        for i, value in enumerate(self.loader):
            self.optimizer.zero_grad()
            index = i + 1
            inp, mask, target, weekday,time_vec,exp,exp_mask,poi= value

            #print(inp.size())
            #print(mask.size())
            result = self.model(inp,mask,time_vec,poi)
            #result = self.model(inp, mask)
            #print(target.size())
            #print(result.size())


            loss = self.ml_criterion(result, torch.flatten(target))
            loss.backward()
            self.optimizer.step()

            average_class_loss += +loss.item()

            #self.optimizer1.step()
            #break

            if index % self._print_every == 0:
                print( f" average_class_loss {average_class_loss / self._print_every}" )
                average_class_loss = 0
        return loss