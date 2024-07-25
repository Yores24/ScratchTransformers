import torch
import torch.nn as nn
import math

class InputEmbeddings(nn.Module):

    def __init__(self,d_model:int,vocab_size:int):
        # d_model is the dimension of the model
        super().__init__()
        self.d_model=d_model
        self.vocab_size=vocab_size
        self.embedding=nn.Embedding(vocab_size,d_model)

    
    def forward(self,x):
        return self.embedding(x)*math.sqrt(self.d_model)
    

class PositionalEncoding(nn.Module):

    def __init__(self,d_model:int,seq_len:int,dropout:float)->None:
        super().__init__()
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
        super().__init__()
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
        super().__init__()
        self.linear_1=nn.Linear(d_model,d_ff) #W1 and B1
        self.dropout=nn.Dropout(dropout)
        self.linear_2=nn.Linear(d_ff,d_model) #W2 and B2

    def forward(self,x):
        # (batch,seq_len,d_model)-->(batch,seq_len,d_ff)-->(batch,seq_len,d_model)
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))


                             

class MutiHeadAttentionBloack(nn.Module):

    def __init__(self, d_model:int,h:int,dropout:float) -> None:
        super().__init__()
        self.d_model=d_model
        self.h=h
        assert d_model%h == 0, "d_model is not divisible by h"
        self.dk=d_model//h
        self.w_q=nn.Linear(d_model,d_model)
        self.w_k=nn.Linear(d_model,d_model)
        self.w_v=nn.Linear(d_model,d_model)

        self.w_o=nn.Linear(d_model,d_model)
        self.dropout=nn.Dropout(dropout)
    @staticmethod
    # what is a static method-mtlb ki hum iss function ko kahi se bhi call krskte h without creating a instance
    # we can just say MultiHeadAttentionBlock.attention()
    def attention(Qp,Vp,Kp,mask,dropout:nn.Dropout):
        dk=Qp.shape[-1]
        attention_scores=(Qp @ Kp.transpose(-2,-1))/math.sqrt(dk)
        if mask is not None:
            attention_scores.masked_fill_(mask==0,-1e9)
        # Yaha masking lagare taki jab softmax lge isme toh woh un values ko nullify krde

        attention_scores=attention_scores.softmax(dim=-1) #(Batch,h,seq_len,seqlen)

        if dropout is not None:
            attention_scores=dropout(attention_scores)
        return (attention_scores @ Vp),attention_scores



    def forward(self,q,k,v,mask):
        Qp=self.w_q(q) # (batch,seqlen,d_model)-->(batch,seqlen,d_model)
        Kp=self.w_k(k) # (batch,seqlen,d_model)-->(batch,seqlen,d_model)
        Vp=self.w_v(v) # (batch,seqlen,d_model)-->(batch,seqlen,d_model)
        # Yaha prr humne multiply krra h

        # (batch,seqlen,d_model)-->(batch,seqlen,h,dk)-->(batch,h,seqlen,dk)
        # Yaha prr humne alag alag usme divide krdiya h
        Qp=Qp.view(Qp.shape[0],Qp.shape[1],self.h,self.dk).transpose(1,2)
        Vp=Vp.view(Vp.shape[0],Vp.shape[1],self.h,self.dk).transpose(1,2)
        Kp=Kp.view(Kp.shape[0],Kp.shape[1],self.h,self.dk).transpose(1,2)

        x,self.attention_scores=MutiHeadAttentionBloack.attention(Qp,Kp,Vp,mask,self.dropout)
        # (batch,h,seq_len,dk )-->(batch,seq_len,h,dk)-->(batch,seqlen,dmodel)
        x=x.transpose(1,2).contiguous().view(x.shape[0],-1,self.h*self.dk)


        # (batch,seqlen,dmodel)-->(batch,seqlen,dmodel)
        return self.w_o(x)
    
class ResidualConnection(nn.Module):

    def __init__(self,dropout:float)->None:

        super().__init__()
        self.dropout=nn.Dropout(dropout)
        self.norm=LayerNormalization()
# Yaha hum add norm layer lere h kyuki hum issi m apni previous layer bhejre
    def forward(self,x,sublayer):
# over hear we can see ki hum jo humare previous layer h sublayer usko hum usse add krdere h
        return x+self.dropout(sublayer(self.norm(x)))
    
    

# creating the encoder block




