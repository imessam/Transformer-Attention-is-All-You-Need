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




def readTextFromFolder(folder_path,limit = 5e5):
    
    lines = []
    
    with open(folder_path,"r", encoding="utf8") as file:
        lines = file.read().split("\n")
        
    if len(lines) > limit:
        lines = lines[:int(limit)]
        
    return lines



def extractTextFromFolders(folders_map,data_map,val_split = 1, limit = 5e5):
        
    for lang in tqdm(data_map.keys(),"extracting text from folders ..."):
        for split in data_map[lang].keys():
            if split == "val":
                continue
            data_map[lang][split] = readTextFromFolder(os.path.join(folders_map[split],f"{split}.{lang}"),limit = limit)
            
        data_map[lang]["val"] = data_map[lang]["train"][int(len(data_map[lang]["train"])*val_split):]
        data_map[lang]["train"] = data_map[lang]["train"][:int(len(data_map[lang]["train"])*val_split)]
        
    gc.collect()    
    return data_map


def extractTokens(data_text,preprocessor):
    
    data_tokens = data_text
    
    for lang in tqdm(data_tokens.keys(),"extracting tokens ..."):
        for split in data_tokens[lang].keys():
            data_tokens[lang][split] = preprocessor.preprocess(data_text[lang][split],lang,mode = 0)
    gc.collect()        
    return data_tokens



def extractEncodings(data_tokens,preprocessor):
    
    data_encodings = data_tokens
    
    for lang in tqdm(data_encodings.keys(),"extracting encodings ..."):
        for split in data_encodings[lang].keys():
            data_encodings[lang][split] = preprocessor.preprocess(data_tokens[lang][split],lang,mode = 1)
    gc.collect()        
    return data_encodings




def translationLoss(output, target, pad_index = 2, label_smoothing = 0.0):
    
    

    loss_func = torch.nn.CrossEntropyLoss(ignore_index    = pad_index,
                                             label_smoothing = label_smoothing)
    vocab_size = output.shape[-1]
    output = output.reshape(-1, vocab_size)
    target = target.reshape(-1).long()
    
    loss = loss_func(output, target)
    
    return loss
    
    
    
def train_model(model, trainLoaders, lossFn, optimizer, pad_index = 2, label_smoothing = 0.0,
                scheduler = None, num_epochs=1, device = "cpu", 
                isSave = False, filename = "transformer-weights", verbose = True):
    
    since = datetime.datetime.now()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float('inf')
    prev_train_loss = 0
    prev_val_loss = 0


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
                   
            if phase == "val":
                
                print(f"{phase} prev epoch Loss: {prev_val_loss}")
                prev_val_loss = epoch_loss
                
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
    model.load_state_dict(best_model_wts)
      
    return model
    

    
    
def evaluate_model(model, test_data, preprocessor, device = "cpu"):
    since = datetime.datetime.now()
    
    model.eval()   # Set model to evaluate mode

    results = []
    # Iterate over data.
    for i,data in enumerate(tqdm(test_data,"Predicting ...")):

        inputs = data["input"]
        labels = data["target"]

        # forward
        outputs =  model(inputs, labels)
    
        result = score(outputs, labels, preprocessor, kind = "bleu")
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



def score(outputs, labels, preprocessor, kind = "bleu"):

    labels = preprocessor.decode(labels["target_encodings"].tolist(),
                                      unpad = True,
                                      idx2word = preprocessor.idx2word_de)

    outputs = preprocessor.decode(torch.argmax(outputs,2).tolist(),
                                      unpad = True,
                                      idx2word = preprocessor.idx2word_de)
    new_labels = []
    for label in labels:
        new_labels.append([label])
    
    if kind == "bleu":
        return corpus_bleu(new_labels, outputs)
    else:
        return None


def infer(model, input_text, preprocessor, eos_idx = 1, device = "cpu"):
    
    model.eval()

    input_tokens = preprocessor.preprocess([input_text], mode = 2).to(device)
    # print(f"input_tokens : {input_tokens.shape}")
    # print(f"input_tokens : {input_tokens}")
    
    print(f"Input English Sentence : {preprocessor.decode(input_tokens.tolist(), unpad = True, idx2word = preprocessor.idx2word_en)}")
    
    input_masks = subsequent_mask(input_tokens, mode = "input", device = device)
    print(f"input_masks : {input_masks.shape}")
    print(f"input_masks : {input_masks}")
        
    input_encodings = model.encode(input_tokens, input_masks)
    # print(f"input_encodings : {input_encodings.shape}")
    # print(f"input_encodings : {input_encodings}")

    
    outputs_tokens = torch.tensor([[0,55]]).to(device).type_as(input_tokens)
    # print(f"outputs_tokens : {outputs_tokens.shape}")
    # print(f"outputs_tokens : {outputs_tokens}")
    
    
    while((outputs_tokens[:,-1] != eos_idx) and (outputs_tokens.shape[-1] < preprocessor.max_sent_len_de)):
        
        outputs_masks = subsequent_mask(outputs_tokens, mode = "output", device = device)
        # print(f"outputs_masks : {outputs_masks.shape}")
        # print(f"outputs_masks : {outputs_masks}")
        
        decodings  = model.decode(outputs_tokens, input_encodings, input_masks, outputs_masks).to(device)
        # print(f"decodings : {decodings.shape}")
        # print(f"decodings : {decodings}")
        
        output_scores = model.finalLayer(decodings[:,-1])
        # print(f"output_scores : {output_scores.shape}")
        # print(f"output_scores : {output_scores}")
        
        
        output_proba = model.softmax(output_scores)
        # print(f"output_proba : {output_proba.shape}")
        # print(output_proba)
        
        next_word = torch.argmax(output_proba, dim = -1).unsqueeze(0)
        # print(f"next_word : {next_word.shape}")
        
        outputs_tokens = torch.cat([outputs_tokens,next_word],dim = -1)
        # print(f"outputs_tokens : {outputs_tokens.shape}")
        # print(outputs_tokens)
        
        

    return preprocessor.decode(outputs_tokens.tolist(), unpad = False, idx2word = preprocessor.idx2word_de)

