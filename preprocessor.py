import os
import torch
from tqdm.auto import tqdm

class WMTPreProcessor:
    
    
    def __init__(self,vocabPath,mappings_path):
        
        self.vocabPath = vocabPath
        self.mappings_path = mappings_path
        
        self.vocab_en = []
        self.vocab_de = []
        
        self.word2idx_en = {}
        self.word2idx_de = {}
        
        self.idx2word_en = {}
        self.idx2word_de = {}
        
        self.en_de_mappings = {}
        self.de_en_mappings = {}
        
        self.max_sent_len_en = 0
        self.max_sent_len_de = 0
        
        self.START = 0
        self.EOS = 1
        self.PAD = 2
          
    
    
    def initialize(self):
        
        self.vocab_en,self.word2idx_en,self.idx2word_en = self.buildVocab(os.path.join(self.vocabPath,"vocab.en"))
        self.vocab_de,self.word2idx_de,self.idx2word_de = self.buildVocab(os.path.join(self.vocabPath,"vocab.de"))
        
        # self.en_de_mappings,self.de_en_mappings = self.build_mappings(self.mappings_path)
        
        
    
    def buildVocab(self,filePath):
        
        vocab = ["<START>", "<EOS>", "<PAD>"]
        word2idx = {"<START>":self.START, "<EOS>":self.EOS ,"<PAD>":self.PAD}
        idx2word = {self.START:"<START>", self.EOS:"<EOS>" , self.PAD:"<PAD>"}
        
        content = ""

        with open(filePath,"r", encoding="utf8") as file:
            content = file.read()

        content = content.split("\n")
        count = 3
        for word in tqdm((content),"Building Vocab ...."):
            
            word = word.lower()
            
            if word not in word2idx:
                
                vocab.append(word)
                word2idx[word] = count
                idx2word[count] = word
                
                count += 1

        return vocab,word2idx,idx2word
    
    
    
    def build_mappings(self,filePath):

        en_de_mappings = {}
        de_en_mappings = {}

        content = ""

        with open(filePath,"r", encoding="utf8") as file:
            content = file.read()

        content = content.split("\n")

        for line in tqdm(content,"Building mappings ...."):

            if len(line) > 0: 
                en,de,val = tuple(line.split(" "))
                en,de = en.lower(),de.lower()

                if en not in en_de_mappings:
                    en_de_mappings[en] = []
                if de not in de_en_mappings:
                    de_en_mappings[de] = []

                en_de_mappings[en].append({"de":de,"val":val})
                de_en_mappings[de].append({"en":en,"val":val})

        return en_de_mappings,de_en_mappings

    
    
    
    
    def tokenize(self,sentences):
        tokens = []
        
        for sentence in tqdm(sentences,"tokenizing ..."):
            sentence_token = ["<START>"] + sentence.lower().split(" ") + ["<EOS>"]
            tokens.append(sentence_token)
        
        return tokens
    

    def encode(self,tokens,word2idx):

        encodings = []

        for token in tqdm(tokens,"encoding ...."):
            
            encoding = []
            
            for word in token:
                encoding.append(word2idx.get(word,word2idx["<unk>"]))
                            
            encodings.append(encoding)

        return encodings
    
    
    
    def clean(self,tokens):

        cleaned_tokens = []

        for i,token in enumerate(tokens):
            # remove punctuation from each word
            stripped = [w.translate(self.table) for w in token]
            # remove remaining tokens that are not alphabetic
            new_token = [word for word in stripped if word.isalpha()]

            cleaned_tokens.append(new_token)

        return cleaned_tokens
    
    
    def pad(self,encodings,word2idx,max_sent_len):
        
        padded_encodings = encodings.copy()
        
        for encoding in tqdm(padded_encodings,"padding ..."):
            for _ in range(len(encoding),max_sent_len):
                encoding.append(word2idx["<PAD>"])
                
        return torch.tensor(padded_encodings)
    

    def decode(self,encodings,unpad,idx2word):

        tokens = []

        for encoding in tqdm(encodings,"decoding ..."):
            
            words = []
            
            for idx in encoding:
                if unpad and idx == self.PAD:
                    break
                words.append(idx2word.get(idx))
                if idx == self.EOS:
                  break
                
            tokens.append(words)

        return tokens
    
    
    
    def set_max_sent_len(self,encodings,lang = "en"):
        
        max_sent_len = self.max_sent_len_en if lang == "en" else self.max_sent_len_de
        
        for encoding in tqdm(encodings,"setting max sent len ...."):
            max_sent_len = max(max_sent_len,len(encoding))
        
        if lang == "en":
            self.max_sent_len_en = max_sent_len
        else:
            self.max_sent_len_de = max_sent_len
            
        return max_sent_len
    
    
    def preprocess(self,sentences,lang = "en",mode = 0):
    
        mapper = self.word2idx_en if lang == "en" else self.word2idx_de
        max_sent_len = self.max_sent_len_en if lang == "en" else self.max_sent_len_de
        
        if mode == 0:
            tokens = self.tokenize(sentences)
            max_sent_len = self.set_max_sent_len(tokens,lang)
            return tokens
            
        elif mode == 1:
            encodings = self.encode(sentences,mapper)
            padded_encodings = self.pad(encodings,mapper,max_sent_len)
            
            return padded_encodings
        
        else :
            
            tokens = self.tokenize(sentences)
            encodings = self.encode(tokens,mapper)
            padded_encodings = self.pad(encodings,mapper,max_sent_len)
            
            return padded_encodings
            

    
    
    
    
