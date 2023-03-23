import torch.nn as nn 
import torch 
from torchvision import models
import config 
import torch.nn.functional as F 
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
            # backbone,  
            # bipartite_matcher, 
            num_queries, 
            d_model,
            num_patches,
            num_head, 
            num_encoders, 
            num_decoders, 
            dropout, 
            backbone_out_features=2048,
            train_backbone=False, 
            num_classes=91
        ): 
        super().__init__() 
        model = models.resnet50(weights="IMAGENET1K_V1", progress=True)
        backbone = torch.nn.Sequential(*(list(model.children())[:-2]))

        if not train_backbone:
            print("Not training the backbone ") 
            for param in backbone.parameters():
                param.requires_grad = False

        self.d_model = d_model
        self.backbone = backbone
        self.transformer = transformer = Transformer(num_queries, d_model, num_patches, num_head, num_encoders, num_decoders, dropout)
        self.matcher = None
        self.feature_projection =  nn.Conv2d(backbone_out_features, d_model, kernel_size=1) # used to project the features to a new space of dimension d_model
        self.boxe_head = BoxesHead(d_model, d_model, 4) 
        self.label_head = LabelHead(d_model, num_classes + 1)

    def forward(self, x): 
        bs = x.shape[0]
        features = self.feature_projection(self.backbone(x)) # (bs, c, p, p) 
        # reshape 
        features = features.view(bs, self.d_model, -1)
        out = self.transformer(features)

        label_pred = self.label_head(out) 
        boxes_pred = self.boxe_head(out).sigmoid() # scale between 0 and 1

        num_queries = label_pred.shape[0] 
        num_classes = label_pred.shape[-1]

        # reshape outputs 
        label_pred = label_pred.permute((1, 0, 2))
        boxes_pred = boxes_pred.permute((1, 0, 2))

        return {
            "labels" : label_pred, 
            "boxes" : boxes_pred
        } 


    

class Transformer(nn.Module): 
    """
    Transformer architecture from the DETR paper. 
    The class will receive an encoder module and a decoder module 

     
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

    def __init__(
            self, 
            num_queries, 
            d_model, 
            num_patches, 
            num_head, 
            num_encoders, 
            num_decoders, 
            dropout
        ):
        super().__init__() 
        self.num_patches = num_patches
        self.num_queries = num_queries
        self.queries_embedding = nn.Embedding(num_queries, d_model)
        self.pos_embedding = nn.Embedding(num_patches, d_model)
        self.encoder = TransformerEncoder(d_model, num_head, dropout, num_encoders)
        self.decoder = TransformerDecoder(d_model, num_head, dropout, num_decoders)

        
    def forward(self, x): 
        """
        Attributes
        ----------
        x : tensor (bs, c, num_pathes) 
            feature maps
        """
        bs = x.shape[0]
        spatial_encoding = self.pos_embedding(torch.arange(self.num_patches, device=x.device))[None, :, :].repeat(bs, 1, 1) # (bs, num_patches, d_model)
        object_queries = self.queries_embedding(torch.arange(self.num_queries, device=x.device))[None, :, :].repeat(bs, 1, 1) # (bs, 100, d_model)

        # reshape for multihead 
        spatial_encoding = torch.permute(spatial_encoding, (1, 0, 2)) 
        object_queries = torch.permute(object_queries, (1, 0, 2)) 
        x = torch.permute(x, (2, 0, 1))

        input_decoder = torch.zeros_like(object_queries)
        out_encoder = self.encoder(x, spatial_encoding) 
        out_decoder = self.decoder(input_decoder, object_queries, out_encoder, spatial_encoding)     

        return out_decoder
         


class EncoderBlock(nn.Module): 

    def __init__(self, d_model, num_head, dropout): 
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, num_head, dropout) 
        self.gelu = nn.GELU() 
        self.dropout1 = nn.Dropout()
        self.dropout2 = nn.Dropout()
        self.layer_norm1 = nn.LayerNorm(d_model) 
        self.layer_norm2 = nn.LayerNorm(d_model) 
        self.ffn1 = nn.Linear(d_model, d_model)
        self.ffn2 = nn.Linear(d_model, d_model)

    def forward(self, x, pos_embedding): 
        # normalization of the input 
        x_norm = self.layer_norm1(x)
        q = x_norm + pos_embedding 
        k = x_norm + pos_embedding
        out1 = self.self_attn(query=q, key=k, value=x_norm)[0] # return the output only
        out1 = self.dropout1(out1) + x
        out2 = self.ffn2(self.gelu(self.ffn1(self.layer_norm2(out1))))
        out2 = self.dropout2(out2) + out1
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
        self.dropout1 = nn.Dropout() 
        self.dropout2 = nn.Dropout()
        self.ffn1 = nn.Linear(d_model, d_model)
        self.ffn2 = nn.Linear(d_model, d_model)
        self.gelu = nn.GELU() 

    def forward(self, x, object_queries, memory, pos_embedding): 
        x_norm = self.layer_norm1(x)
        k = q = x + object_queries 
        v = x_norm
        out1 = self.self_attn(q, k, v)[0] 
        out1 = self.dropout1(out1) + x_norm 
        out2 = self.ffn2(self.gelu(self.ffn1(self.layer_norm2(out1)))) 
        out2 = self.dropout2(out2) + out1 
        return out2



class TransformerDecoder(nn.Module): 

    def __init__(self, d_model, num_head, dropout, num_decoders): 
        super().__init__() 
        self.layers = nn.ModuleList([DecoderBlock(d_model, num_head, dropout) for _ in range(num_decoders)])
    
    def forward(self, x, object_queries, memory, pos_embedding): 
        out = x 
        for layer in self.layers: 
            out = layer(out, object_queries, memory, pos_embedding)
        return out 


class BoxesHead(nn.Module): 

    def __init__(self, in_d, hidden_d, out_d): 
        super().__init__()
        self.linear1 = nn.Linear(in_d, hidden_d) 
        self.linear2 = nn.Linear(hidden_d, hidden_d) 
        self.linear3 = nn.Linear(hidden_d, out_d)

        self.dropout1 = nn.Dropout()
        self.dropout2 = nn.Dropout() 
        self.relu = nn.ReLU()
 

    def forward(self, x): 
        x = self.dropout1(self.relu(self.linear1(x))) 
        x = self.dropout2(self.relu(self.linear2(x))) 
        out = self.linear3(x) 
        return out 
    

class LabelHead(nn.Module): 

    def __init__(self, in_d, num_classes): 
        super().__init__() 
        self.linear = nn.Linear(in_d, num_classes) 

    def forward(self, x): 
        return self.linear(x)
    