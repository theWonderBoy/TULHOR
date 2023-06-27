import time
from datetime import datetime
from pathlib import Path

import torch

from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from dataset import IMDBBertDataset
from Bert import BERT

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")

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


class BertTrainer:

    def __init__(self,
                 model: BERT,
                 dataset: IMDBBertDataset,
                 log_dir: Path,
                 checkpoint_dir: Path = None,
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

        self.loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)

        #self.writer = SummaryWriter(str(log_dir))
        self.checkpoint_dir = checkpoint_dir

        self.ml_criterion = nn.NLLLoss(ignore_index=0).to(device)
        #self.ml_criterion_spatial = nn.MSELoss(reduction = 'sum').to(device)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        #self.optimizer1 = torch.optim.Adam(multimodel.parameters(), lr=learning_rate)
        #self.multimodelTask = multimodel
        #, weight_decay=0.05
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
            self.save_checkpoint(self.current_epoch, step=-1, loss=0)
            #print(torch.cuda.memory_allocated(device))
            

    def train(self, epoch: int):
        print(f"Begin epoch {epoch}")
        spatial_target_embeddings = self.dataset.geo_embeddings.to(device)
        prev = time.time()
        average_spatial_loss = 0
        average_mlm_loss = 0
        #print(torch.cuda.memory_allocated(0)/1024/1024/1024)
        for i, value in enumerate(self.loader):
            #print(value)
            index = i + 1
            inp, mask, inverse_token_mask, token_target, nsp_target, spatial_mask_target, week_day,time_vec = value
            #print(spatial_mask_target)
            self.optimizer.zero_grad()
            #self.optimizer1.zero_grad()
            #print(token_target)
            #print(token_target)
            #spatial_target = spatial_target_embeddings(spatial_mask_target)
            token = self.model(inp, mask,time_vec)
            #print(spatial_tokens),spatial_tokens
            """
            print("spatial_target size = {0}".format(spatial_target.size()))
            print("spatial_tokens size = {0}".format(spatial_tokens.size()))
            
            print("token size = {0}".format(token.size()))
            print("spatial_tokens size = {0}".format(token.size()))
            print("inverse_token_mask = {0}".format(inverse_token_mask.size()))
            """
            
            tm = inverse_token_mask.unsqueeze(-1).expand_as(token)
            #spatial_tokens_tm = inverse_token_mask.unsqueeze(-1).expand_as(spatial_tokens)
            #print("spatial_tokens_tm size = {0}".format(spatial_tokens_tm.size()))
            #print(token)
            token = token.masked_fill(tm, 0)
            #spatial_tokens= spatial_tokens.masked_fill(spatial_tokens_tm, 0)
            #print(spatial_tokens)
            #print(spatial_target)
            #print(token)
            
            # 1D tensor as target is required
            #print(spatial_tokens)
            #print(spatial_target)
            #print(spatial_tokens)
            #print(spatial_target)
            #loss_token = self.ml_criterion(token.transpose(1, 2), token_target)
            """
            loss_token.backward()
            self.optimizer.step()
            self.model.embedding.requires_grad=False
            self.optimizer.zero_grad()
            loss_token.backward().detach()
            """
            #loss= self.multimodelTask(token.transpose(1, 2),spatial_tokens,token_target,spatial_target )
            #print("token target size = {0}".format(token_target.size()))
            #print("token size = {0}".format(token.transpose(1, 2).size()))
            #print(token_target)
            #print(inp)
            #loss.backward()
            #self.optimizer.step()
            #self.optimizer1.step()
            #self.model.embedding.requires_grad=True
            #loss_nsp = self.criterion(nsp, nsp_target)
            loss_token = self.ml_criterion(token.transpose(1, 2), token_target)
            #loss_token_spatial =  ( self.ml_criterion_spatial(spatial_tokens, spatial_target) / torch.count_nonzero(token_target) )
            #losses = [loss_token,loss_token_spatial] # a list of per-task losses
            loss_token.backward()
             # calculate the gradient can apply gradient modification
            self.optimizer.step()  # apply gradient step
            #loss_token = self.ml_criterion(token.transpose(1, 2), token_target)
            #loss_token_spatial =  ( self.ml_criterion_spatial(spatial_tokens, spatial_target) / torch.count_nonzero(token_target) )
            

            #self.optimizer.step()
            #self.optimizer1.step()
            #loss_token_spatial =  ( self.ml_criterion_spatial(spatial_tokens, spatial_target) / torch.count_nonzero(token_target) )
            #loss = loss_token + loss_token_spatial
            #loss = [loss_token,loss_token_spatial]
            average_spatial_loss += 0
            average_mlm_loss += loss_token.item()

            #self.optimizer1.step()
            #break

            if index % self._print_every == 0:
                elapsed = time.gmtime(time.time() - prev)
                s = self.training_summary(elapsed, index, average_spatial_loss, average_mlm_loss)
                if index % self._accuracy_every == 0:
                    s += self.accuracy_summary(index, token, [], token_target, nsp_target,inverse_token_mask)
                print(s)

                average_spatial_loss = 0
                average_mlm_loss = 0
        return loss_token

    def training_summary(self, elapsed, index, average_spatial_loss, average_mlm_loss):
        passed = percentage(self.batch_size, self._ds_len, index)
        global_step = self.current_epoch * len(self.loader) + index

        print_spatial_loss = average_spatial_loss / self._print_every
        print_mlm_loss = average_mlm_loss / self._print_every

        s = f"{time.strftime('%H:%M:%S', elapsed)}"
        s += f" | Epoch {self.current_epoch + 1} | {index} / {self._batched_len} ({passed}%) | " \
              f"MLM loss {print_mlm_loss:6.2f} |  spatial_loss {print_spatial_loss:6.2f}"

        #self.writer.add_scalar("NSP loss", print_nsp_loss, global_step=global_step)
        #self.writer.add_scalar("MLM loss", print_mlm_loss, global_step=global_step)
        #self.writer.add_scalar("spatial loss", print_mlm_loss, global_step=global_step)
        return s

    def accuracy_summary(self, index, token, nsp, token_target, nsp_target, inverse_token_mask):
        global_step = self.current_epoch * len(self.loader) + index
        #nsp_acc = nsp_accuracy(nsp, nsp_target)
        token_acc = token_accuracy(token, token_target, inverse_token_mask)

        #self.writer.add_scalar("NSP train accuracy", nsp_acc, global_step=global_step)
        #self.writer.add_scalar("Token train accuracy", token_acc, global_step=global_step)

        return f" | Token accuracy {token_acc}"

    def save_checkpoint(self, epoch, step, loss):
        if not self.checkpoint_dir:
            return

        prev = time.time()
        name = f"bert_epoch{epoch}_step{step}_{datetime.utcnow().timestamp():.0f}.pt"
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict()
        }, self.checkpoint_dir)
       
        print()
        print('=' * self._splitter_size)
        print(f"Model saved as '{name}' for {time.time() - prev:.2f}s")
        print('=' * self._splitter_size)
        print()

    def load_checkpoint(self, path: Path):
        print('=' * self._splitter_size)
        print(f"Restoring model {path}")
        checkpoint = torch.load(path)
        self.current_epoch = checkpoint['epoch']
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print("Model is restored.")
        print('=' * self._splitter_size)