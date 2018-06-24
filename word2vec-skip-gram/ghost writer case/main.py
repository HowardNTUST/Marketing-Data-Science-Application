# Importing the libraries
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import matplotlib.pyplot as plt
from torch.nn import functional as F
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
USE_CUDA = torch.cuda.is_available()
gpus = [0]
torch.cuda.set_device(gpus[0])

FloatTensor = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if USE_CUDA else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if USE_CUDA else torch.ByteTensor

import skip_model
train_data,word2index, unigram_table,vocab = skip_model.SkipGram(data = None, columns = 'comment', cut = True, WINDOW_SIZE = 2, dataname = 'newdata_pytorch.csv')



class SkipgramNegSampling(nn.Module):
    
    def __init__(self, vocab_size, projection_dim):
        super(SkipgramNegSampling, self).__init__()
        self.embedding_v = nn.Embedding(vocab_size, projection_dim) # center embedding
        self.embedding_u = nn.Embedding(vocab_size, projection_dim) # out embedding
        self.logsigmoid = nn.LogSigmoid()
                
        initrange = (2.0 / (vocab_size + projection_dim))**0.5 # Xavier init
        self.embedding_v.weight.data.uniform_(-initrange, initrange) # init
        self.embedding_u.weight.data.uniform_(-0.0, 0.0) # init
        
    def forward(self, center_words, target_words, negative_words):
        center_embeds = self.embedding_v(center_words) # B x 1 x D
        target_embeds = self.embedding_u(target_words) # B x 1 x D
        
        neg_embeds = -self.embedding_u(negative_words) # B x K x D
        
        positive_score = target_embeds.bmm(center_embeds.transpose(1, 2)).squeeze(2) # Bx1
        negative_score = torch.sum(neg_embeds.bmm(center_embeds.transpose(1, 2)).squeeze(2), 1).view(negs.size(0), -1) # BxK -> Bx1
        
        loss = self.logsigmoid(positive_score) + self.logsigmoid(negative_score)
        
        return -torch.mean(loss)
    
    def prediction(self, inputs):
        embeds = self.embedding_v(inputs)
        
        return embeds

EMBEDDING_SIZE = 300
BATCH_SIZE = 256
EPOCH = 100
NEG = 10 # Num of Negative Sampling
losses = []
model = SkipgramNegSampling(len(word2index), EMBEDDING_SIZE)
if USE_CUDA:
    model = model.cuda()
    
optimizer = optim.Adam(model.parameters(), lr=0.001)

model
model.state_dict()

for epoch in range(EPOCH):
    for i,batch in enumerate(skip_model.getBatch(BATCH_SIZE, train_data)):
        
        inputs, targets = zip(*batch)
        
        inputs = torch.cat(inputs) # B x 1
        targets = torch.cat(targets) # B x 1
        negs = skip_model.negative_sampling(targets, unigram_table, NEG)
        model.zero_grad()

        loss = model(inputs, targets, negs)
        
        loss.backward()
        optimizer.step()
    
        losses.append(loss.data.tolist())
        
    if epoch % 5 == 0:
        print("Epoch : %d, mean_loss : %.02f" % (epoch, np.mean(losses)))
        losses = []
#        torch.save(model, 'skipgram.pt')
        torch.cuda.empty_cache()

# test sim
skip_model.word_similarity('滋潤',vocab, USE_CUDA=True,word2index=word2index,model=model,how_many = 10)