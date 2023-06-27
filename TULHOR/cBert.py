import torch
import gensim
from torch import nn
import torch.nn.functional as f
import math
from Bert import BERT
class BERTForclassification(nn.Module):

    def __init__(self, Bert, user_size,dim_inp, dim_out):
        super(BERTForclassification, self).__init__()


        #self.sub = BERTWithout_head(vocab_size,dim_inp,dim_out, attention_heads=4)
        #print(dir(Bert))
        self.embedding = Bert.embedding
        self.encoder = Bert.encoder
        self.encoder1 = Bert.encoder1
        self.encoder2 = Bert.encoder2
        self.encoder3 = Bert.encoder3

        self.user_classification_layer = nn.Linear(dim_inp, user_size)
        self.softmax = nn.LogSoftmax(dim=-1)


    def forward(self, input_tensor: torch.Tensor, attention_mask: torch.Tensor,time=None, poi_tensor=None):
        #print("reach here")
        #embedded = self.embedding(input_tensor)
        embedded,pure_embed = self.embedding(input_tensor,time,poi_tensor)
        encoded = self.encoder(pure_embed, attention_mask,embedded[0],embedded[1],embedded[2],embedded[3])
        #encoded = self.encoder1(encoded, attention_mask,embedded[0],embedded[1],embedded[2],embedded[3])
        #encoded = self.encoder1(encoded, attention_mask,embedded[0],embedded[1],embedded[2])
        #encoded = self.encoder1(encoded,attention_mask,embedded)
        #encoded = self.encoder2(encoded, attention_mask,embedded)
       # encoded = self.encoder3(encoded,attention_mask,embedded)

        user_classifications = self.user_classification_layer(encoded[:,0])

        #first_word = encoded[:, 0, :]
        return user_classifications