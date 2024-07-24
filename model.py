import torch
import torch.nn as nn
import math

class InputEmbeddings(nn.Module):

    def __init__(self,d_model:int,vocab_size:int):
        # d_model is the dimension of the model
        super().__init__
        self.d_model=d_model
        self.vocab_size=vocab_size
        self.embedding=nn.Embedding(vocab_size,d_model)

    
    def forward(self,x):
        return self.embedding(x)*math.sqrt(self.d_model)
    

class PositionalEncoding(nn.Module):

    def __init__(self,d_model:int,seq_len:int,dropout:float)->None:
        super().__init__
        self.d_model=d_model
        self.seq_len=seq_len
        self.dropout=nn.Dropout(dropout)

        # create a matrix of size(seq_len,d_model)

        posm=torch.zeros(seq_len,d_model)

        # create a matx for representing the position of word in the sentence

        pos=torch.arange(0,seq_len,dtype=torch.float).unsqueeze(1)
        # shape is (seq_len,1)
        # refer notes for the formula but yahi formula h Positional encoding ka

        divt=torch.exp(torch.arange(0,d_model,2).float()*(-math.log(10000.0)/d_model))
        
        # apply the sin to even positions

        posm[:,0::2]=torch.sin(pos*divt)

        # apply the cos to odd positions
        posm[:,1::2]=torch.cos(pos*divt)

        # adding batch dim to tensor mtlb 2d se 3d krre
        posm=posm.unsqueeze(0) #ab dim hogyi(1,seq_len,d_model)

        # register to buffer to the model
        # what is buffer - when you want to keep a tensor not as a param but as a file the you register it in the buffer

        self.register_buffer("pe",posm)

    # adding the positional encoding to words in sentence

    def forword(self,x):
        x=x+(self.posm[:,:x.shape[1],:]).requires_grad(False)
        return self.dropout(x)


class LayerNormalization(nn.Module):

    def __init__(self,eps:float=10**-6 ) -> None:
        super().__init__
        self.eps=eps
        # nn.Parameter makes the parameter learnable
        self.alpha=nn.Parameter(torch.ones(1)) #multiplied
        self.beta=nn.Parameter(torch.zeros(1)) #added

    def forward(self,x):
        mean=x.mean(dim=-1,keepdim=True)

        std=x.std(dim=-1,keepdim=True)
        return self.alpha*(x-mean)/(std*self.eps)+self.beta
    

class FeedForwardBlock(nn.Module):

    def __init__(self, d_model:int,d_ff:int,dropout:float) -> None:
        super().__init__
        self.linear_1=nn.Linear(d_model,d_ff) #W1 and B1
        self.dropout=nn.Dropout(dropout)
        self.linear_2=nn.Linear(d_ff,d_model) #W2 and B2

    def forward(self,x):
        # (batch,seq_len,d_model)-->(batch,seq_len,d_ff)-->(batch,seq_len,d_model)
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))


                             

    def