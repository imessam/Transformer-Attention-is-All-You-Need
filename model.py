import torch
import math
from torch import nn
from torchvision import transforms



class Transformer(torch.nn.Module):
    
    def __init__(self, src_vocabSize, tgt_vocabSize, src_max_len = 150, tgt_max_len = 150, noHeads = 8, d_model = 512, d_ff = 2048, 
                 dropout = 0.1, noEncoder = 6, noDecoder = 6, pad_index = 2, device ="cpu"):

        super(Transformer, self).__init__()
        self.device = device
        self.d_model = d_model
        
#         ##Input Embedding##
#         self.input_embedding = torch.nn.Embedding(num_embeddings = src_vocabSize, embedding_dim = d_model, padding_idx = pad_index, device = device)

#         ##Output Embedding##
#         self.output_embedding = torch.nn.Embedding(num_embeddings = tgt_vocabSize, embedding_dim = d_model, padding_idx = pad_index, device = device)

        ##Input and Output Embedding##
        self.embedding = torch.nn.Embedding(num_embeddings = src_vocabSize, embedding_dim = d_model, padding_idx = pad_index, device = device)

        ##Positional Encoding##
        self.inp_pos_encoding = PositionalEncoding(d_model, dropout, max_len = src_max_len, device = device)
        self.out_pos_encoding = PositionalEncoding(d_model, dropout, max_len = tgt_max_len, device = device)
        
        ##Encoder##
        self.encoder = torch.nn.ModuleList([EncoderLayer(noHeads, d_model, d_ff, dropout, device = device) for i in range(noEncoder)])
        
        ##Decoder##
        self.decoder = torch.nn.ModuleList([DecoderLayer(noHeads, d_model, d_ff, dropout, device = device) for i in range(noDecoder)])

        ##Final Layer with shared weights##
        self.finalLayer = torch.nn.Linear(in_features = d_model, out_features = tgt_vocabSize, device = device)   
        self.finalLayer.weight = nn.Parameter(self.embedding.weight)
        self.softmax = torch.nn.Softmax(dim = -1)
        
        
    def encode(self, input_pos_embeddings, inputs_masks):
        
        
        ##Encoder forward##
        temp = input_pos_embeddings
        for layer in self.encoder:
            temp = layer(temp, inputs_masks)
        input_encodings = temp
        # print(f"input_encodings : {input_encodings.shape}")

        return input_encodings


    def decode(self, output_pos_embeddings, input_encodings, inputs_masks, outputs_masks):

        ##Decoder Forward##
        temp = output_pos_embeddings
        for layer in self.decoder:
            temp = layer(temp, input_encodings, inputs_masks, outputs_masks)
        decodings = temp
        # print(f"decodings : {decodings.shape}")

        return decodings

        
    def forward(self, inputs, outputsShifted):
        
        
        inputs_tokens, inputs_masks = inputs["encodings"].to(self.device), inputs["masks"].to(self.device)
        outputs_tokens, outputs_masks = outputsShifted["decoder_input_encodings"].to(self.device), outputsShifted["masks"].to(self.device)
        
        # print(f"inputs_tokens : {inputs_tokens.shape}")
        # print(f"inputs_masks : {inputs_masks.shape}")
        # print(f"outputs_tokens : {outputs_tokens.shape}")
        # print(f"outputs_masks : {outputs_masks.shape}")
        
        ##Input embeddings##
        # input_embeddings = self.input_embedding(inputs_tokens) * math.sqrt(self.d_model)
        input_embeddings = self.embedding(inputs_tokens) * math.sqrt(self.d_model)
        # print(f"input embeddings : {input_embeddings.shape}")

        ##Add Positional Encoding##
        input_pos_embeddings = self.inp_pos_encoding(input_embeddings)
        # print(f"input_pos_embeddings : {input_pos_embeddings.shape}")
        
        ##Output embeddings##
        # output_embeddings = self.output_embedding(outputs_tokens) * math.sqrt(self.d_model)
        output_embeddings = self.embedding(outputs_tokens) * math.sqrt(self.d_model)
        # print(f"output embeddings : {output_embeddings.shape}")

        ##Add Positional Encoding##
        output_pos_embeddings = self.out_pos_encoding(output_embeddings)
        # print(f"output_pos_embeddings : {output_pos_embeddings.shape}")
        
        
        ##Encoder forward##
        input_encodings = self.encode(input_pos_embeddings, inputs_masks)
        # print(f"input_encodings : {input_encodings.shape}")
        
        ##Decoder Forward##
        decodings = self.decode(output_pos_embeddings, input_encodings, inputs_masks, outputs_masks)
        # print(f"decodings : {decodings.shape}")

        ##Final Probabilities##
        output_proba = self.finalLayer(decodings)
        # print(f"output_proba : {output_proba.shape}")

        return output_proba
    
    

class EncoderLayer(torch.nn.Module):
    
    
    def __init__(self, noHeads, d_model, d_ff, dropout, device ="cpu"):

        super(EncoderLayer, self).__init__()
        
        self.subLayer_1 = torch.nn.ModuleList([MultiHeadAttention(noHeads = noHeads, d_model = d_model, device = device),
                           nn.Dropout(dropout),
                           AddNorm(d_model, device = device)])
        self.subLayer_2 = torch.nn.ModuleList([FeedForward(d_model, d_ff, device = device),
                           nn.Dropout(dropout),
                           AddNorm(d_model, device = device)])
        
        

    def forward(self, input_pos_embeddings, inputs_masks):
        
        ##SubLayer 1 forward##
        att_weights = self.subLayer_1[0](key = input_pos_embeddings,
                                         query = input_pos_embeddings,
                                         value = input_pos_embeddings,
                                         masks = inputs_masks)
        # print(f"att_weights : {att_weights.shape}")
        dropout_att_weights = self.subLayer_1[1](att_weights)
        # print(f"dropout_att_weights : {dropout_att_weights.shape}")
        normalized_att_weights = self.subLayer_1[2](dropout_att_weights, input_pos_embeddings)
        # print(f"normalized_att_weights : {normalized_att_weights.shape}")

        ##SubLayer 2 forward##
        projected_att_weights = self.subLayer_2[0](normalized_att_weights)
        # print(f"projected_att_weights : {projected_att_weights.shape}")
        dropout_att_weights = self.subLayer_2[1](projected_att_weights)
        # print(f"dropout_att_weights : {dropout_att_weights.shape}")
        encodings = self.subLayer_2[2](dropout_att_weights, normalized_att_weights)
        # print(f"encodings : {encodings.shape}")

        return encodings

        
    
    
class DecoderLayer(torch.nn.Module):
    
    
    def __init__(self, noHeads, d_model, d_ff, dropout, device ="cpu"):

        super(DecoderLayer, self).__init__()
        
        self.subLayer_1 = torch.nn.ModuleList([MultiHeadAttention(noHeads = noHeads, d_model = d_model, device = device),
                           nn.Dropout(dropout),
                           AddNorm(d_model, device = device)])
        self.subLayer_2 = torch.nn.ModuleList([MultiHeadAttention(noHeads = noHeads, d_model = d_model, device = device),
                           nn.Dropout(dropout),
                           AddNorm(d_model, device = device)])
        self.subLayer_3 = torch.nn.ModuleList([FeedForward(d_model, d_ff, device = device),
                           nn.Dropout(dropout),
                           AddNorm(d_model, device = device)])
        
        

    def forward(self, output_pos_embeddings, input_encodings, inputs_masks, outputs_masks):

        
        ##SubLayer 1 forward##
        att_weights = self.subLayer_1[0](key = output_pos_embeddings,
                                         query = output_pos_embeddings,
                                         value = output_pos_embeddings,
                                         masks = outputs_masks)
        # print(f"att_weights : {att_weights.shape}")
        dropout_att_weights = self.subLayer_1[1](att_weights)
        # print(f"dropout_att_weights : {dropout_att_weights.shape}")
        normalized_masked_att_weights = self.subLayer_1[2](dropout_att_weights, output_pos_embeddings)
        # print(f"normalized_masked_att_weights : {normalized_masked_att_weights.shape}")

        ##SubLayer 2 forward##
        att_weights = self.subLayer_2[0](key = input_encodings,
                                         query = normalized_masked_att_weights,
                                         value = input_encodings,
                                         masks = inputs_masks)
        # print(f"att_weights : {att_weights.shape}")
        dropout_att_weights = self.subLayer_2[1](att_weights)
        # print(f"dropout_att_weights : {dropout_att_weights.shape}")
        normalized_att_weights = self.subLayer_2[2](dropout_att_weights, normalized_masked_att_weights)
        # print(f"normalized_att_weights : {normalized_att_weights.shape}")

        ##SubLayer 3 forward##
        projected_att_weights = self.subLayer_3[0](normalized_att_weights)
        # print(f"projected_att_weights : {projected_att_weights.shape}")
        dropout_att_weights = self.subLayer_3[1](projected_att_weights)
        # print(f"dropout_att_weights : {dropout_att_weights.shape}")
        decodings = self.subLayer_3[2](dropout_att_weights, normalized_att_weights)
        # print(f"decodings : {decodings.shape}")

        return decodings
    
    
    
class MultiHeadAttention(torch.nn.Module):
    
    
    def __init__(self, noHeads = 8, d_model = 512, device ="cpu"):

        super(MultiHeadAttention, self).__init__()

        self.d_v, self.d_k, self.noHeads = d_model // noHeads, d_model // noHeads, noHeads
        
        self.att = Attention()
        self.linearLayers = torch.nn.ModuleList([torch.nn.Linear(in_features = d_model, out_features = d_model, device = device), 
                              torch.nn.Linear(in_features = d_model, out_features = d_model, device = device), 
                              torch.nn.Linear(in_features = d_model, out_features = d_model, device = device)])
        self.finalLinear = torch.nn.Linear(in_features = noHeads*self.d_v, out_features = d_model, device = device)
        

    def forward(self, key, query, value, masks = None):

        nbatches = key.shape[0]
        
        ##Key, Query, Value projections##
        key_transf = self.linearLayers[0](key).view(nbatches, -1, self.noHeads, self.d_k).transpose(1, 2)
        query_transf = self.linearLayers[1](query).view(nbatches, -1, self.noHeads, self.d_k).transpose(1, 2)
        value_transf = self.linearLayers[2](value).view(nbatches, -1, self.noHeads, self.d_k).transpose(1, 2)

#         print(f"key_transf : {key_transf.shape}")
#         print(f"query_transf : {query_transf.shape}")
#         print(f"value_transf : {value_transf.shape}")

        if masks is not None:
            # Same mask applied to all h heads.
            masks = masks.unsqueeze(1)

        ##Projected Key, Query, Value attention weights##
        attWeight = self.att(key_transf, query_transf, value_transf, masks)
        # print(f"attWeight : {attWeight.shape}")
            
        ##Concatenate attention heads##
        concatHeads = attWeight.transpose(1, 2).contiguous().view(nbatches, -1, self.noHeads * self.d_k)
        # print(f"concatHeads : {concatHeads.shape}")

        ##Project concatenated heads##
        finalProjection = self.finalLinear(concatHeads)
        # print(f"finalProjection : {finalProjection.shape}")

        return finalProjection

        
                          
    
    
class Attention(torch.nn.Module):
    
    
    def __init__(self, device ="cpu"):

        super(Attention, self).__init__()
        
        self.softmax = torch.nn.Softmax(dim = -1)
        

    def forward(self, key, query, value, masks = None):
        
        d_k = key.shape[3]
        scores = torch.matmul(query ,torch.transpose(key,2,3))
        # print(f"scores : {scores.shape}")
        
        scaled_scores = scores/math.sqrt(d_k)
        # print(f"scaled_scores : {scaled_scores.shape}")
        
        if masks is not None:
            # print(f"masks : {masks.shape}")
            scaled_scores = scaled_scores.masked_fill(masks == 0,-1e9)
            # print(f"masked scaled_scores : {scaled_scores.shape}")
        
        normalized_scores = self.softmax(scaled_scores)
        # print(f"normalized_scores : {normalized_scores.shape}")
        
        att_weights = torch.matmul(normalized_scores , value)
        # print(f"att_weights : {att_weights.shape}")
        

        return att_weights
         
          
    
    
    
class AddNorm(torch.nn.Module):
    
    
    def __init__(self, d_model, device ="cpu"):

        super(AddNorm, self).__init__()
        
        self.layerNorm = LayerNorm(d_model, device = device)
        

    def forward(self, in1, in2):
        
        return self.layerNorm(in1+in2)


class LayerNorm(torch.nn.Module):
    
    
    def __init__(self, d_model, eps=1e-6, device ="cpu"):

        super(LayerNorm, self).__init__()
        
        self.a_2 = nn.Parameter(torch.ones(d_model)).to(device)
        self.b_2 = nn.Parameter(torch.zeros(d_model)).to(device)
        self.eps = eps
        

    def forward(self, x):
        
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2  
    
    
class FeedForward(torch.nn.Module):
    
    
    def __init__(self, d_model, d_ff, device ="cpu"):

        super(FeedForward, self).__init__()

        self.w1 = torch.nn.Linear(in_features = d_model, out_features = d_ff, device = device)
        self.w2 = torch.nn.Linear(in_features = d_ff, out_features = d_model, device = device)
        

    def forward(self, x):
        
                
        return self.w2(torch.relu(self.w1(x)))



class PositionalEncoding(torch.nn.Module):
    
    
    def __init__(self, d_model, dropout, max_len=5000, device ="cpu"):

        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)
        

    def forward(self, x):
        
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)


    
