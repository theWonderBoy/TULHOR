import torch
import gensim
from torch import nn
import torch.nn.functional as f
import math
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class JointEmbedding(nn.Module):

    def __init__(self, vocab_size, size, poi_size,spath):
        super(JointEmbedding, self).__init__()

        self.size = size

        model = gensim.models.Word2Vec.load("Spatial-Extractor/" + spath)
        weights = torch.FloatTensor(torch.cat( ( torch.FloatTensor(model.wv.vectors),torch.zeros(5, 512) )))
        self.s_emb = nn.Embedding.from_pretrained(weights)
        self.s_emb.requires_grad = True
        #torch.FloatTensor(model.wv.vectors)
        self.spatial_emb =  nn.Embedding(vocab_size, size)
        self.spatial_emb.requires_grad = True
        self.poi_emb =  nn.Embedding(poi_size, size)
        #self.spatial_emb.requires_grad = True
        self.mlp = nn.Linear(size+64, size)
        self.time_emb= TemporalEncoding(size)
        #print(size)
       # self.segment_emb = nn.Embedding(vocab_size, size)
        self.norm = nn.LayerNorm(size)

    def forward(self, input_tensor,time_input,poi_input):
        #sentence_size = input_tensor.size(-1)
        pos_tensor = self.attention_position(self.size, poi_input)

        #segment_tensor = torch.zeros_like(input_tensor).to(device)
        #segment_tensor[:, sentence_size // 2 + 1:] = 1
        time_emb = self.time_emb(time_input)
        #output = pos_tensor + time_emb + self.s_emb(input_tensor)
        
        
        return (( self.norm(pos_tensor),self.norm(time_emb),self.norm(self.s_emb(input_tensor)),self.norm(self.poi_emb(poi_input)) ) , self.norm(self.spatial_emb(input_tensor)) )

    def attention_position(self, dim, input_tensor):
        batch_size = input_tensor.size(0)
        sentence_size = input_tensor.size(-1)

        pos = torch.arange(sentence_size, dtype=torch.long).to(device)
        d = torch.arange(dim, dtype=torch.long).to(device)
        d = (2 * d / dim)

        pos = pos.unsqueeze(1)
        pos = pos / (1e4 ** d)

        pos[:, ::2] = torch.sin(pos[:, ::2])
        pos[:, 1::2] = torch.cos(pos[:, 1::2])

        return pos.expand(batch_size, *pos.size())

    def numeric_position(self, dim, input_tensor):
        pos_tensor = torch.arange(dim, dtype=torch.long).to(device)
        return pos_tensor.expand_as(input_tensor)

class TemporalEncoding(nn.Module):
    def __init__(self, embed_size):
        super().__init__()
        self.w = nn.Parameter((torch.from_numpy(1 / 10 ** np.linspace(0, 9, embed_size))).float(), requires_grad=True)
        self.b = nn.Parameter(torch.zeros(embed_size).float(), requires_grad=True)
        self.div = math.sqrt(1. / embed_size)

    def forward(self, kwargs):
        timestamp = kwargs  # (batch, seq_len)
        time_encode = torch.cos(timestamp.unsqueeze(-1) * self.w.reshape(1, 1, -1) + self.b.reshape(1, 1, -1))
        return self.div * time_encode
class AttentionHead1(nn.Module):

    def __init__(self, dim_inp, dim_out,pureEmbed=None):
        super(AttentionHead1, self).__init__()

        self.dim_inp = dim_inp

        self.q = nn.Linear(dim_inp, dim_out)
        self.k = nn.Linear(dim_inp, dim_out)
        self.v = nn.Linear(dim_inp, dim_out)
        self.fusion = nn.Linear(dim_inp*5, dim_inp)
    def forward(self, input_tensor: torch.Tensor, attention_mask: torch.Tensor = None,integerated_emb=None, time_emb=None, spatial_emb=None,poi_emb=None):
        if integerated_emb!=None:
          value = self.v(input_tensor)
          #print(value.size())
          integerated_emb = torch.cat((input_tensor,integerated_emb,time_emb,spatial_emb,poi_emb),2)
          #print(integerated_emb.size())
          integerated_emb = self.fusion(integerated_emb)
          #print(integerated_emb.size())
          query, key = self.q(integerated_emb), self.k(integerated_emb)
        else:
          query, key, value = self.q(input_tensor), self.k(input_tensor), self.v(input_tensor)

        scale = query.size(1) ** 0.5
        scores = torch.bmm(query, key.transpose(1, 2)) / scale

        scores = scores.masked_fill_(attention_mask, -1e9)
        attn = f.softmax(scores, dim=-1)
        context = torch.bmm(attn, value)

        return context


class MultiHeadAttention(nn.Module):

    def __init__(self, num_heads, dim_inp, dim_out):
        super(MultiHeadAttention, self).__init__()

        self.heads = nn.ModuleList([
            AttentionHead1(dim_inp, dim_out) for _ in range(num_heads)
        ])
        self.linear = nn.Linear(dim_out * num_heads, dim_inp)
        self.norm = nn.LayerNorm(dim_inp)

    def forward(self, input_tensor: torch.Tensor, attention_mask: torch.Tensor,pureEmbed=None,time_emb=None, spatial_emb=None,poi_emb=None):
        s = [head(input_tensor, attention_mask,pureEmbed,time_emb,spatial_emb,poi_emb) for head in self.heads]
        scores = torch.cat(s, dim=-1)
        scores = self.linear(scores)
        return self.norm(scores)


class Encoder(nn.Module):

    def __init__(self, dim_inp, dim_out, attention_heads=4, dropout=0.1):
        super(Encoder, self).__init__()

        self.attention = MultiHeadAttention(attention_heads, dim_inp, dim_out)  # batch_size x sentence size x dim_inp
        self.feed_forward = nn.Sequential(
            nn.Linear(dim_inp, dim_out),
            nn.Dropout(dropout),
            nn.GELU(),
            nn.Linear(dim_out, dim_inp),
            nn.Dropout(dropout)
        )
        self.norm = nn.LayerNorm(dim_inp)

    def forward(self, input_tensor: torch.Tensor, attention_mask: torch.Tensor,pureEmbed=None,time_emb=None, spatial_emb=None,poi_emb=None):
        context = self.attention(input_tensor, attention_mask,pureEmbed,time_emb,spatial_emb,poi_emb)
        res = self.feed_forward(context)
        return self.norm(res)


class BERT(nn.Module):

    def __init__(self, vocab_size, dim_inp, dim_out, attention_heads=4 , poi_input= 10000 , spath=None):
        super(BERT, self).__init__()


        #self.sub = BERTWithout_head(vocab_size,dim_inp,dim_out, attention_heads=4)
        self.embedding = JointEmbedding(vocab_size, dim_inp,poi_input,spath)
        self.encoder = Encoder(dim_inp, dim_out, attention_heads)
        self.encoder1 = Encoder(dim_inp, dim_out, attention_heads)
        self.encoder2 = Encoder(dim_inp, dim_out, attention_heads)
        self.encoder3 = Encoder(dim_inp, dim_out, attention_heads)
        self.token_prediction_layer = nn.Linear(dim_inp, vocab_size)
        #self.spatial_prediction_layer = nn.Linear(dim_inp+dim_inp, 2)
        self.softmax = nn.LogSoftmax(dim=-1)



    def forward(self, input_tensor: torch.Tensor, attention_mask: torch.Tensor,time_tensor, poi_tensor):
        #print("reach here")
        comb_embd,pure_embed = self.embedding(input_tensor,time_tensor,poi_tensor)
        encoded = self.encoder(pure_embed, attention_mask,comb_embd[0],comb_embd[1],comb_embd[2],comb_embd[3])
        #encoded = self.encoder1(encoded, attention_mask,comb_embd[0],comb_embd[1],comb_embd[2])
        #encoded = self.encoder2(encoded,attention_mask,comb_embd)
        #encoded = self.encoder3(encoded,attention_mask,comb_embd)

        token_predictions = self.token_prediction_layer(encoded)
        #spatial_token_predictions = self.spatial_prediction_layer(encoded)

        #first_word = encoded[:, 0, :]
        return self.softmax(token_predictions) #self.classification_layer(first_word)