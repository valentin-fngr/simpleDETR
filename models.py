import torch.nn as nn 
import torch 

## TODO : 
## take care of shapes for attention modules !!!



class DETR(nn.Module): 
    """
    DETR main class. It uses an encoder and decoder module to compute the final set of predictions. 

    Attributes
    ----------
    num_queries : int
                Number of output queries, number of output sets. The paper uses N=100.
    transformer : Transformer (nn.Module) 
                A nn.Module Transformer that represents the architecture of the DETR paper.
    bipartite_matcher : nn.Module
                A nn.Module class that will compuyte the optimized set assignement (described as theta in the paper).
    """

    def __init__(
            self, 
            num_queries, 
            transformer, 
            bipartite_matcher
        ): 
        super().__init__() 

    def forward(self): 
        pass






class Transformer(nn.Module): 
    """
    Transformer architecture from the DETR paper. 
    The class will receive an encoder module and a decoder module 

    Attributes
    ----------
    queries_embedding : nn.Embedding 
                Set embedding matrix for end to end learning
    positional_embedding : nn.Embeeding 
                Positional embedding in decoder 
    encoder : nn.Module 
                Encoder module of the DETR architecture
    decoder : nn.Module 
                Decoder module of the DETR architecture
    dropout : float 
                Dropout value for both encoder and decoder 
    """

    def __init__(self, num_queries, d_model, num_patches, encoder, decoder, dropout):
        super().__init__() 
        self.queries_embedding = nn.Embedding(num_queries, d_model)
        self.pos_embedding = nn.Embedding(num_patches, d_model)
        self.encoder = encoder 
        self.decoder = decoder
        
    def forward(self, x): 
        pass 



class EncoderBlock(nn.Module): 

    def __init__(self, d_model, num_head, dropout): 
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, num_head, dropout) 
        self.gelu = nn.GELU() 
        self.dropout1 = nn.Dropout()
        self.layer_norm1 = nn.LayerNorm(d_model) 
        self.layer_norm2 = nn.LayerNorm(d_model) 
        self.ffn = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, pos_embedding): 
        # normalization of the input 
        x_norm = self.layer_norm1(x)
        q = x_norm + pos_embedding 
        k = x_norm + pos_embedding
        out1 = self.self_attn(query=q, key=k, value=x_norm)[0] # return the output only
        out1 = self.dropout1(out1) + x
        out2 = self.gelu(self.ffn(out1))  
        out2 = self.layer_norm2(self.dropout(out2) + out1)
        
        return out2 


class TransformerEncoder(nn.Module): 

    def __init__(self, d_model, num_head, dropout, num_encoders): 
        super().__init__() 
        self.layers = nn.ModuleList([EncoderBlock(d_model, num_head, dropout) for _ in range(num_encoders)])

    def forward(self, x, pos_embedding): 
        
        out = x 
        for layer in self.layers: 
            out = layer(x, pos_embedding) 
        return out





class DecoderBlock(nn.Module): 

    def __init__(self, d_model, num_head, dropout): 
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, num_head, dropout)
        self.multi_head_attn = nn.MultiheadAttention(d_model, num_head, dropout) 
        self.layer_norm1 = nn.LayerNorm(d_model) 
        self.layer_norm2 = nn.LayerNorm(d_model) 
        self.layer_norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout() 
        self.dropout2 = nn.Dropout()
        self.ffn = nn.Linear(d_model, d_model)
        self.gelu = nn.GELU() 

    def forward(self, x, object_queries, memory, pos_embedding): 

        x_norm = self.layer_norm1(x)
        k = q = x + object_queries 
        v = x_norm
        out1 = self.self_attn(q, k, v)[0] 
        out1 = self.dropout1(out1) + x_norm
        out2 = self.gelu(self.ffn(out1)) 
        out2 = self.layer_norm2(self.dropout2(out2) + out1)



class DecoderTransformer(nn.Module): 

    def __init__(self, d_model, num_head, dropout, num_decoders): 
        super().__init__() 
        self.layers = nn.ModuleList([DecoderBlock(d_model, num_head, dropout) for _ in range(num_decoders)])
    
    def forward(self, x, object_queries, memory, pos_embedding): 

        out = x 
        for layer in self.layers: 
            out = layer(out, object_queries, memory, pos_embedding)
        return out 



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
encoder = TransformerEncoder(192, 8, 0.1, 5)
print(encoder)
decoder = DecoderTransformer(192, 8, 0.1, 5)
print(decoder)