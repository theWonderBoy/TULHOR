import time
from datetime import datetime
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import numpy
import pandas as pd
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from TULDataset import TULDataset
from TUL import TUL
from tqdm import tqdm

class TulTrainer:

    def __init__(self,
                 model: TUL,
                 dataset: TULDataset,
                 print_progress_every: int = 10,
                 print_accuracy_every: int = 50,
                 batch_size: int = 24,
                 learning_rate: float = 0.005,
                 epochs: int = 5
                 ):
        self.model = model
        self.dataset = dataset

        self.batch_size = batch_size
        self.epochs = epochs
        self.current_epoch = 0
        df = dataset.df
        user_list = df['User'].drop_duplicates().values.tolist()
        print(f'The number of users {len(user_list)}')
        train_indices, test_indices = [] ,[]
        for user in tqdm(user_list):
          temp = df.loc[df['User'] == user].index
          train_sample = temp[:int(len(temp)*0.8)]
          test_sample = temp[int(len(temp)*0.8):]
          train_indices = train_indices + list(train_sample)
          test_indices = test_indices + list(test_sample)
        print(f'Number of training samples = {len(train_indices)}  \n  Number of testing samples ={len(test_indices)} ')
        train = torch.utils.data.Subset(self.dataset, train_indices )
        self.loader = DataLoader(train, batch_size=self.batch_size, shuffle=True)

        self.test = torch.utils.data.Subset(self.dataset, test_indices )
        self.test_loader = DataLoader(self.test, batch_size=1, shuffle=True)

        self.ml_criterion = nn.CrossEntropyLoss().to(device)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self._splitter_size = 35

        self._ds_len = len(self.dataset)
        self._batched_len = self._ds_len // self.batch_size

        self._print_every = print_progress_every
        self._accuracy_every = print_accuracy_every

    def calculate_test_accuracy(self):
        self.model.eval()
        top1,top3,top5 = 0,0,0
        targets,results = [],[]
        with torch.no_grad():
          for i , value in enumerate(self.test_loader):
            inp, target, un_padded_length = value
            result = self.model(inp,un_padded_length)
            targets.append(target.item())
            top1 += self.find_k_accuracy(1,result,target)
            results.append(torch.topk(result, 1).indices.item())
            top3 += self.find_k_accuracy(3,result,target)
            top5 += self.find_k_accuracy(5,result,target)

        top1 = top1/len(self.test)
        top3 = top3/len(self.test)
        top5 = top5/len(self.test)
        f1_score_macro = f1_score(targets, results, average='macro')
        r_score = recall_score(targets, results, average='macro')
        p_score = precision_score(targets, results, average='macro')
        print(f"top@1 = {top1} | top@3 = {top3} | top@5 = {top5} | macro-P={p_score} | macro-R{r_score} |f1_score_macro={f1_score_macro}" )
        self.model.train()
    def __call__(self):
        for self.current_epoch in range(self.current_epoch, self.epochs):
            #print(torch.cuda.memory_allocated(0)/1024/1024/1024)
            loss = self.train(self.current_epoch)
            self.calculate_test_accuracy()
            #print(torch.cuda.memory_allocated(device))
    
    def find_k_accuracy(self,k,result,target):
      #print(result)
      #print(target)
      if target in torch.topk(result, k).indices:
        return 1
      return 0    
    def train(self, epoch: int):
        print(f"Begin epoch {epoch}")
        prev = time.time()
        classfication_loss = 0
        for i, value in enumerate(self.loader):
            self.optimizer.zero_grad()
            index = i + 1
            inp, user, un_padded_length = value
            results = self.model(inp,un_padded_length)
            #print(user)
            #print(results)
            class_loss = self.ml_criterion(results, torch.flatten(user))
           
            class_loss.backward()
             # calculate the gradient can apply gradient modification
            self.optimizer.step()  # apply gradient step
           
            classfication_loss += class_loss.item()

            if index % self._print_every == 0:
              print(f'classfication_loss = {classfication_loss/self._print_every}')
              classfication_loss = 0
        return class_loss
