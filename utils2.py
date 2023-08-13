import os
import gc
import torch
import datetime
import copy
from copy import deepcopy
from time import sleep
from operator import itemgetter
from tqdm.auto import tqdm
from nltk.translate.bleu_score import corpus_bleu
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing




def init_bpe_tokenizer():
    
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    trainer = BpeTrainer(special_tokens=["[CLS]","[SEP]","[UNK]","[PAD]", "[MASK]", "[BOS]", "[EOS]"])
    tokenizer.pre_tokenizer = Whitespace()
    
    return tokenizer, trainer


def process_tokenizer(tokenizer):
    
    tokenizer.post_processor = TemplateProcessing(
    single="[BOS] $A [EOS]",
    pair="[CLS] $A [SEP] $B:1 [SEP]:1",
    special_tokens=[
        ("[CLS]", tokenizer.token_to_id("[CLS]")),
        ("[SEP]", tokenizer.token_to_id("[SEP]")),
        ("[UNK]", tokenizer.token_to_id("[UNK]")),
        ("[PAD]", tokenizer.token_to_id("[PAD]")),
        ("[MASK]", tokenizer.token_to_id("[MASK]")),
        ("[BOS]", tokenizer.token_to_id("[BOS]")),
        ("[EOS]", tokenizer.token_to_id("[EOS]"))
    ],
    )
    
    return tokenizer
    

def create_train_bpe_tokenizer(folders_map, data_map, tokenizers_path):
    
    tokenizer = None
    
    print(os.path.exists(os.path.join(tokenizers_path,f"tokenizer_shared.json")),os.path.join(tokenizers_path,f"tokenizer_shared.json"))
    if os.path.exists(os.path.join(tokenizers_path,f"tokenizer_shared.json")):
        print("loading pretrained  ...." + os.path.join(tokenizers_path,f"tokenizer_shared.json"))
        tokenizer = Tokenizer.from_file(os.path.join(tokenizers_path,f"tokenizer_shared.json"))
    else:
        tokenizer, trainer = init_bpe_tokenizer()
        for lang in tqdm(data_map.keys(),"training tokenizer per language ..."):

            files = []
            for split in data_map[lang].keys():
                if split == "val":
                    continue
                files.append(os.path.join(folders_map[split],f"{split}.{lang}"))

            tokenizer.train(files, trainer)

        tokenizer = process_tokenizer(tokenizer)
        tokenizer.enable_padding(pad_id = tokenizer.token_to_id("[PAD]"), pad_token = "[PAD]")
        tokenizer.save(os.path.join(tokenizers_path,f"tokenizer_shared.json"))
    
    return tokenizer
    
def create_train_bpe_tokenizer_old(folders_map,data_map, tokenizers_path):
    
    tokenizers = {}
    for lang in tqdm(data_map.keys(),"training tokenizer per language ..."):
        print(os.path.exists(os.path.join(tokenizers_path,f"tokenizer-{lang}.json")),os.path.join(tokenizers_path,f"tokenizer-{lang}.json"))
        if os.path.exists(os.path.join(tokenizers_path,f"tokenizer-{lang}.json")):
            print("loading pretrained  ...." + os.path.join(tokenizers_path,f"tokenizer-{lang}.json"))
            tokenizer = Tokenizer.from_file(os.path.join(tokenizers_path,f"tokenizer-{lang}.json"))
        else:
            files = []
            tokenizer, trainer = init_bpe_tokenizer()
            for split in data_map[lang].keys():
                if split == "val":
                    continue
                files.append(os.path.join(folders_map[split],f"{split}.{lang}"))

            tokenizer.train(files, trainer)
            tokenizer.save(os.path.join(tokenizers_path,f"tokenizer-{lang}.json"))
        tokenizer = process_tokenizer(tokenizer)
        tokenizer.enable_padding(pad_id = tokenizer.token_to_id("[PAD]"), pad_token = "[PAD]")
        tokenizers[lang] = (tokenizer)

    
    return tokenizers




def readTextFromFolder(folder_path,limit = 5e5):
    
    lines = []
    
    with open(folder_path,"r", encoding="utf8") as file:
        lines = file.read().split("\n")
        
    if len(lines) > limit:
        lines = lines[:int(limit)]
        
    return lines



def extractTextFromFolders(folders_map, data_map, val_split = 1, limit = 5e5):
    
    data_text = deepcopy(data_map)
    
    for lang in tqdm(data_text.keys(),"extracting text from folders ..."):
        for split in data_text[lang].keys():
            if split == "val":
                continue
            data_text[lang][split] = readTextFromFolder(os.path.join(folders_map[split],f"{split}.{lang}"),limit = limit)
            
        data_text[lang]["val"] = data_text[lang]["train"][int(len(data_text[lang]["train"])*val_split):]
        data_text[lang]["train"] = data_text[lang]["train"][:int(len(data_text[lang]["train"])*val_split)]
        
    gc.collect()    
    return data_text



def extractTokens(data_text,tokenizer):
    
    data_tokens = data_text
    
    for lang in tqdm(data_tokens.keys(),"extracting tokens ..."):
        for split in data_tokens[lang].keys():
            if split  != "train":
                max_len = len(data_tokens[lang]["train"][0])
                print(max_len, len(data_tokens[lang]["train"][0]))
                tokenizer.enable_padding(length = max_len, pad_id = tokenizer.token_to_id("[PAD]"), pad_token = "[PAD]")
                tokenizer.enable_truncation(max_length = max_len)
            data_tokens[lang][split] = tokenizer.encode_batch(data_text[lang][split])
    gc.collect()        
    return data_tokens


def extractTokens_old(data_text,tokenizers):
    
    data_tokens = data_text
    
    for lang in tqdm(data_tokens.keys(),"extracting tokens ..."):
        tokenizer = tokenizers[lang]
        for split in data_tokens[lang].keys():
            if split  != "train":
                max_len = len(data_tokens[lang]["train"][0])
                tokenizer.enable_padding(length = max_len, pad_id = tokenizer.token_to_id("[PAD]"), pad_token = "[PAD]")
                tokenizer.enable_truncation(max_length = max_len)
            data_tokens[lang][split] = tokenizer.encode_batch(data_text[lang][split])
    gc.collect()        
    return data_tokens



def extractEncodings(data_tokens):
    
    data_encodings = data_tokens
    
    for lang in tqdm(data_encodings.keys(),"extracting encodings ..."):
        for split in data_encodings[lang].keys():
            data_encodings[lang][split] = torch.tensor([encoding.ids for encoding in  data_encodings[lang][split]])
    gc.collect()        
    return data_encodings




def translationLoss(output, target, pad_index = 2, label_smoothing = 0.1):
    
    

    loss_func = torch.nn.CrossEntropyLoss(ignore_index    = pad_index,
                                             label_smoothing = label_smoothing)
    vocab_size = output.shape[-1]
    output = output.reshape(-1, vocab_size)
    target = target.reshape(-1).long()
    
    loss = loss_func(output, target)
    
    return loss
    
    
    
def train_model(model, trainLoaders, lossFn, optimizer, pad_index = 2, label_smoothing = 0.1,
                scheduler = None, num_epochs=1, device = "cpu", 
                isSave = False, filename = "transformer-weights", verbose = True):
    
    since = datetime.datetime.now()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float('inf')
    prev_train_loss = 0
    prev_val_loss = 0

    train_losses = []
    val_losses = []


    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        epoch_loss = 0
        
        for phase in ['train','val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0

            # Iterate over data.
            for i,data in enumerate(tqdm(trainLoaders[phase],"Predicting ...")):

                if device != "cpu":
                    torch.cuda.empty_cache()
                
                inputs = data["input"]
                labels = data["target"]
                inputs["encodings"], inputs["masks"] = inputs["encodings"].to(device), inputs["masks"].to(device)
                labels["decoder_input_encodings"], labels["target_encodings"], labels["masks"] = (
                    labels["decoder_input_encodings"].to(device),labels["target_encodings"].to(device), labels["masks"].to(device))
    

                # zero the parameter gradients
                optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase == 'train'):

                    # forward
                    outputs =  model(inputs, labels)            
                    
                    loss = lossFn(outputs, labels["target_encodings"], pad_index = pad_index, label_smoothing = label_smoothing)
                    
                    if phase == "train":
                        loss.backward()
                        optimizer.step()
                        
                        # learning rate scheduler
                        if scheduler is not None:
                            scheduler.step()

                # statistics
                running_loss += (loss.item()*len(inputs))
                if verbose:
                    print(f' Iteration Loss : {loss.item()*len(inputs)}, lr = {optimizer.param_groups[0]["lr"]}')
            
            epoch_loss = running_loss / len(trainLoaders[phase])
            
            if phase == "train":
                            
                print(f"{phase} prev epoch Loss: {prev_train_loss}")
                prev_train_loss = epoch_loss
                train_losses.append(epoch_loss)
                   
            if phase == "val":
                
                print(f"{phase} prev epoch Loss: {prev_val_loss}")
                prev_val_loss = epoch_loss
                val_losses.append(epoch_loss)
                
                # deep copy the model
                if epoch_loss<best_loss:
                    best_loss = epoch_loss
                    best_model_wts = copy.deepcopy(model.state_dict())
                    if isSave:
                        torch.save(model.state_dict(), f"trained/{filename}")    
                        
            print(f"{phase} current epoch Loss: {epoch_loss}, lr = {optimizer.param_groups[0]['lr']}")


    print()

    time_elapsed = (datetime.datetime.now() - since).total_seconds()
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val loss: {:4f}'.format(best_loss))


    ## load best model weights
    # model.load_state_dict(best_model_wts)

      
    return model, train_losses, val_losses
    

    
    
def evaluate_model(model, test_data, tokenizer, device = "cpu"):
    since = datetime.datetime.now()
    
    model.eval()   # Set model to evaluate mode

    results = []
    # Iterate over data.
    for i,data in enumerate(tqdm(test_data,"Predicting ...")):

        inputs = data["input"]
        labels = data["target"]

        # forward
        outputs =  model(inputs, labels)
    
        result = score(outputs, labels, tokenizer, kind = "bleu")
        results.append(result)

    results = torch.tensor(results)

    print(f" Result : {results.mean()}")

    print()

    time_elapsed = (datetime.datetime.now() - since).total_seconds()
    print('Evaluating complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
        
    return results



def subsequent_mask(tokens, mode, pad_index = 2, device = "cpu"):
    "Mask out subsequent positions."
    size = tokens.size(-1)
    attn_shape = (1, size, size)
    
    mask = (tokens != pad_index).unsqueeze(-2).to(device)
    # print(f"mask : {mask.shape}")
    if mode == "input":
        return mask
    
    subsequent_mask = (torch.triu(torch.ones(attn_shape), diagonal=1).type(
        torch.uint8
        ) == 0).to(device)
    # print(f"subsequent_mask : {subsequent_mask.shape}")
    # print(f"mask & subsequent_mask : {(mask & subsequent_mask).shape}")
    
    return mask & subsequent_mask



def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)



def score(outputs, labels, tokenizer, kind = "bleu"):
    
    labels = tokenizer.decode_batch(labels["target_encodings"].tolist())

    outputs = tokenizer.decode_batch(torch.argmax(outputs,2).tolist())
    
    new_labels = []
    for label in labels:
        new_labels.append([label])
    
    if kind == "bleu":
        return corpus_bleu(new_labels, outputs)
    else:
        return None


def infer(model, input_text, tokenizer, device = "cpu"):
    
    model.eval()
    
    # tokenizer_en, tokenizer_de = tokenizers["en"], tokenizers["de"]
    
    # eos_idx = tokenizer_de.token_to_id("[EOS]")
    eos_idx = tokenizer.token_to_id("[EOS]")
    
    # input_tokens = torch.tensor([tokenizer_en.encode(input_text).ids]).to(device)
    # input_tokens = input_tokens[input_tokens != tokenizer_en.token_to_id("[BOS]")].view((input_tokens.shape[0],input_tokens.shape[1]-1))
    
    input_tokens = torch.tensor([tokenizer.encode(input_text).ids]).to(device)
    input_tokens = input_tokens[input_tokens != tokenizer.token_to_id("[BOS]")].view((input_tokens.shape[0],input_tokens.shape[1]-1))
    # print(f"input_tokens : {input_tokens.shape}")
    # print(f"input_tokens : {input_tokens}")
    
    # print(f"Input English Sentence : {tokenizer_en.decode(input_tokens[0].tolist())}")
    print(f"Input English Sentence : {tokenizer.decode(input_tokens[0].tolist())}")
    
    # input_masks = subsequent_mask(input_tokens, pad_index = tokenizer_en.token_to_id("[PAD]"), mode = "input", device = device)
    input_masks = subsequent_mask(input_tokens, pad_index = tokenizer.token_to_id("[PAD]"), mode = "input", device = device)

    print(f"input_masks : {input_masks.shape}")
    print(f"input_masks : {input_masks}")
        
    input_encodings = model.encode(input_tokens, input_masks)
    # print(f"input_encodings : {input_encodings.shape}")
    # print(f"input_encodings : {input_encodings}")

    
    # outputs_tokens = torch.tensor([[tokenizer_de.token_to_id("[BOS]")]]).to(device).type_as(input_tokens)
    outputs_tokens = torch.tensor([[tokenizer.token_to_id("[BOS]")]]).to(device).type_as(input_tokens)

    # print(f"outputs_tokens : {outputs_tokens.shape}")
    # print(f"outputs_tokens : {outputs_tokens}")
    
    
    while((outputs_tokens[:,-1] != eos_idx) and (outputs_tokens.shape[-1] < 200)):
        
        # outputs_masks = subsequent_mask(outputs_tokens, pad_index = tokenizer_de.token_to_id("[PAD]"), mode = "output", device = device)
        outputs_masks = subsequent_mask(outputs_tokens, pad_index = tokenizer.token_to_id("[PAD]"), mode = "output", device = device)

        # print(f"outputs_masks : {outputs_masks.shape}")
        # print(f"outputs_masks : {outputs_masks}")
        
        decodings  = model.decode(outputs_tokens, input_encodings, input_masks, outputs_masks).to(device)
        # print(f"decodings : {decodings.shape}")
        # print(f"decodings : {decodings}")
        
        output_scores = model.finalLayer(decodings[:,-1])
        # print(f"output_scores : {output_scores.shape}")
        # print(f"output_scores : {output_scores}")
        
        
        output_proba = model.softmax(output_scores)
#         print(f"output_proba : {output_proba.shape}")
#         print(output_proba)
        
        next_word = torch.argmax(output_proba, dim = -1).unsqueeze(0)
        # print(f"next_word : {next_word.shape}")
        
        outputs_tokens = torch.cat([outputs_tokens,next_word],dim = -1)
        # print(f"outputs_tokens : {outputs_tokens.shape}")
        # print(outputs_tokens)
        
        

    # return tokenizer_de.decode(outputs_tokens[0].tolist())
    return tokenizer.decode(outputs_tokens[0].tolist())

