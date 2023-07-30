import os
import torch
from utils2 import *
from torch.utils.data import Dataset


class WMT(Dataset):
    
    def __init__(self,inpt_encodings,tgt_encodings, eos_index = 1, pad_index = 2):
        
        
        self.inpt_encodings = inpt_encodings[:,1:]
        self.dec_inpt_encodings = tgt_encodings[tgt_encodings != eos_index].view((tgt_encodings.shape[0],tgt_encodings.shape[1]-1))
        self.tgt_encodings = tgt_encodings[:,1:]
        
        
        self.pad_index = pad_index
        
        self.inpt_masks = subsequent_mask(self.inpt_encodings, mode = "input", pad_index = self.pad_index)
        self.tgt_masks = subsequent_mask(self.dec_inpt_encodings, mode = "target", pad_index = self.pad_index)

    def __len__(self):
        
        return len(self.inpt_encodings)

    def __getitem__(self, idx):
        
        return {"input":{"encodings":self.inpt_encodings[idx],"masks":self.inpt_masks[idx]},
                "target":{"decoder_input_encodings":self.dec_inpt_encodings[idx],"target_encodings":self.tgt_encodings[idx],"masks":self.tgt_masks[idx]}}
    
    
  
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        