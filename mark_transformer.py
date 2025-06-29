import math
import torch
import torch.nn as nn

# Input embedddings
class InputEmbeddings(nn.Module):
    def __init__(self, d_model:int , vocab_size:int):
        super().__init__()
        self.d_model= d_model
        self.vocab_size= vocab_size
        self.embedding= nn.Embedding(vocab_size, d_model)

    def forward(self,x):
        return self.embedding(x)* math.sqrt(self.d_model)    

# Positional encoding
class PositionalEncoding(nn.Module):
    def __init__(self,d_model:int,seq_len:int,dropout:float)->None:
        super().__init__()
        self.d_model= d_model
        self.seq_len= seq_len
        self.dropout= nn.Dropout(dropout)

        # Create a matrix of shape (seq_len, d_model)
        pe= torch.zeros(seq_len,d_model)
        # Create a vector of shape (seq_len,1)
        position= torch.arange(0, seq_len,dtype=torch.float).unsqueeze(1)
        div_term= torch.exp(torch.arange(0, d_model,2).float()*(math.log(10000.0)/d_model))

        # Apply the sin even positions and cos to odd
        pe[:, 0::2]= torch.sin(position*div_term)
        pe[:, 1::2]= torch.cos(position*div_term)

        pe= pe.unsqueeze(0) #(1, seq_len, d_model)
        self.register_buffer('pe',pe)
    def forward(self, x):
        x= x+ (self.pe[:,: x.shape[1],:]).requires_grad_(False) # tells the model not to learn this since the pos are fixed
        return self.dropout(x)

# Multihead attention
class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, d_model:int,h:int,dropout:float)->None:
        super().__init__()
        self.d_model= d_model
        self.h= h

        assert d_model % h==0, "d_model is not divisible by h"
        self.d_k =d_model //h
        self.w_q= nn.Linear(d_model, d_model) # wq
        self.w_k= nn.Linear(d_model, d_model) # wk
        self.w_v= nn.Linear(d_model, d_model) # wv

        self.w_o= nn.Linear(d_model,d_model) # wo
        self.dropout= nn.Dropout(dropout)
    @staticmethod
    def attention(query,key,value, mask,dropout:nn.Dropout):
        # masking to handle different outputs from encoder and target data
        d_k= query.shape[-1]

        # batch,h,seq_len, d_k --> batch,h, seq_len, seq_len
        attention_score= (query @ key.transpose(-2,-1))/math.sqrt(d_k)
        if mask is not None:
            attention_score.mask_fill_(mask==0,-1e9)
        attention_score= attention_score.softmax(dim=-1) # batch,h seq_len,seq_len
        if dropout is not None:
            attention_score= dropout(attention_score)
        return (attention_score @ value), attention_score

    def forward(self,q,k,v,mask):
        query = self.w_q(q) #(batch,seq_len,d_model) --> (batch,seq_len,d_model)
        key = self.w_k(k) #(batch,seq_len,d_model) --> (batch,seq_len,d_model)
        value = self.w_v(v) #(batch,seq_len,d_model) --> (batch,seq_len,d_model)

        # (batch,h,seq_len,d_k) --> (batch,h,seq_len,d_k)
        query= query.view(query.shape[0],query.shape[1],self.h,self.d_k).transpose(1,2)
        key= key.view(key.shape[0],key.shape[1],self.h,self.d_k).transpose(1,2)
        value= value.view(value.shape[0],value.shape[1],self.h,self.d_k).transpose(1,2)

        x, self.attention_scores= MultiHeadAttentionBlock.attention(query,key,value,mask,self.dropout)

        # (batch,h,seq_len.d_k) --> (batch,seq_len,d_model)
        x= x.transpose(1,2).contiguous().view(x.shape[0],-1,self.h *self.d_k)

        return self.w_o(x) # (batch,seq_len,d_model) -->(batch,seq_len,d_k)
    
# layer Normolization
class LayerNormalization(nn.Module):
    def __int__(self,eps:float= 10**-6)->None:
        super().__init__()
        self.eps= eps
        self.alpha= nn.Parameter(torch.ones(1)) # multiplied
        self.bias= nn.Parameter(torch.zeros(1)) # added
    def forward(self,x):
        mean= x.mean(dim=-1,keepdim=True)
        std= x.std(dim=-1,keepdim=True)
        return self.alpha *(x-mean)/ (std+self.eps)+self.bias

# Feed Forward
class FeedForwardBlock(nn.Module):
    def  __init__(self, d_model:int,d_ff:int,dropout:float)->None:
        super().__init__()
        self.linear_1= nn.Linear(d_model,d_ff) # w1 and b1
        self.dropout= nn.Dropout(dropout)
        self.linear_2= nn.Linear(d_ff,d_model) # w2 and b2

    def forward(self, x):
        # (batch, seq_len, d_model) --> (batch, seq_len,d_model)
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))

# Residial
class ResidualConnection(nn.Module):
    def __init__(self,dropout:float)->None:
        super().__init__()
        self.dropout= nn.Dropout(dropout)
        self.norm= LayerNormalization()
    def forward(self,x,sublayer):
        return x+ self.dropout(sublayer(self.norm()))
    
# Encoder block
class EncoderBlock(nn.Module):
    def __init__(self,self_attention_block:MultiHeadAttentionBlock,feed_forward_block:FeedForwardBlock,dropout:float)->None:
        super().__init__()
        self.self_attention_block= self_attention_block
        self.feed_forward_block= feed_forward_block
        self.residual_connection= nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])

    def forward(self,x,src_mask):
        x= self.residual_connection[0](x, lambda x: self.self_attention_block(x,x,x,src_mask))
        x= self.residual_connection[1](x, self.feed_forward_block)
        return x
    
# Encoder
class Encoder(nn.Module):
    def __init__(self,layers:nn.ModuleList)->None:
        super().__init__()
        self.layers= layers
        self.norm= LayerNormalization()
    def forward(self,x,mask):
        for layer in self.layers:
            x= layer(x,mask)
        return self.norm(x)
# Decoder block
class DecoderBlock(nn.Module):
    def __init__(self,self_attention_block:MultiHeadAttentionBlock,cross_attention_block:MultiHeadAttentionBlock, feed_forword_block:FeedForwardBlock,dropout:float)->None:
        super().__init__()
        self.self_attention_block= self_attention_block
        self.cross_attention_block= cross_attention_block
        self.feed_forward_block= feed_forword_block
        self.residual_connection= nn.ModuleList([ResidualConnection(dropout) for _ in range(3)])

    def forward(self,x,encoder_output, src_mask, tgt_mask):
        x= self.residual_connection[0](x, lambda x: self.self_attention_block(x,x,x,tgt_mask))
        x= self.residual_connection[1](x, lambda x: self.cross_attention_block(x,encoder_output,encoder_output, src_mask))
        x= self.residual_connection[2](x, self.feed_forward_block)
        return x
# Decoder
class Decoder(nn.Module):
    def __init__(self,layers:nn.ModuleList)->None:
        super().__init__()
        self.layers= layers
        self.norm= LayerNormalization()
    def forward(self,x,encoder_output,src_mask,tgt_mask):
        for layer in self.layers:
            x= layer(x,encoder_output,src_mask,tgt_mask)
        return self.norm(x)
# Projection
class ProjectionLayer(nn.Module):
    def __init__(self, d_model: int, vocab_size: int) ->None:
        super().__init__()
        self.proj= nn.Linear(d_model,vocab_size)

    def forward(self,x):
        return torch.log_softmax(self.proj(x),dim=-1)
    

# Transformer
class Transformer(nn.Module):
    def __init__(self,encoder:Encoder,decoder:Decoder,src_embed:InputEmbeddings,tgt_embed:InputEmbeddings,src_pos:PositionalEncoding,tgt_pos:PositionalEncoding,projection_layer:ProjectionLayer)->None:
        super().__init__()
        self.encoder= encoder
        self.decoder= decoder
        self.src_embed= src_embed
        self.tgt_embed= tgt_embed
        self.src_pos= src_pos
        self.tgt_pos= tgt_pos
        self.projection_layer= projection_layer

    def encode(self,src,src_mask):
        src= self.src_embed(src)
        src= self.src_pos(src)
        return self.encoder(src,src_mask)
    
    def decode(self, encoder_output,src_mask,tgt,tgt_mask):
        tgt= self.tgt_embed(tgt)
        tgt= self.tgt_pos(tgt)
        return self.decoder(tgt,encoder_output,src_mask,tgt_mask)

    def project(self,x):
        return self.projection_layer(x)
    
# full transformer
def build_transformer(src_vocab_size:int,tgt_vocab_size:int,src_seq_len:int,tgt_seq_len:int,d_model:int=512, N:int=6,h:int=8,dropout:float=0.1,d_ff:int= 2048)->Transformer:

    # create the embedding layers
    src_embed= InputEmbeddings(d_model,src_vocab_size)
    tgt_embed= InputEmbeddings(d_model,tgt_vocab_size)

    # create the positional encoding layers
    src_pos= PositionalEncoding(d_model,src_seq_len,dropout)
    tgt_pos= PositionalEncoding(d_model,tgt_seq_len,dropout)

    # create the encoder blocks
    encoder_blocks=[]
    for _ in range(N):
        encoder_self_attention_block=MultiHeadAttentionBlock(d_model,h,dropout)
        feed_forward_block= FeedForwardBlock(d_model,d_ff,dropout)
        encoder_block= EncoderBlock(encoder_self_attention_block,feed_forward_block)
        encoder_blocks.append(encoder_block)

    # create the decoder blocks
    decoder_blocks=[]
    for _ in range(N):
        decoder_self_attention_block=MultiHeadAttentionBlock(d_model,h,dropout)
        decoder_cross_attention_block=MultiHeadAttentionBlock(d_model,h,dropout)
        feed_forward_block= FeedForwardBlock(d_model,d_ff,dropout)
        decoder_block= DecoderBlock(decoder_self_attention_block,decoder_cross_attention_block,feed_forward_block)
        decoder_blocks.append(decoder_block)
    
    # create encoder and decoder
    encoder =Encoder(encoder_blocks)
    decoder =Decoder(decoder_blocks)

    # create projection layers
    projection_layer= ProjectionLayer(d_model,tgt_vocab_size)

    # create transformer
    transformer= Transformer(encoder,decoder,src_embed, tgt_embed,src_pos,tgt_pos,projection_layer)

    # initialize the parameters
    for p in transformer.parameters():
        if p.dim() >1:
            nn.init.xavier_uniform_(p)
    
    return transformer