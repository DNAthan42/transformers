import argparse
import os
from tqdm import trange, tqdm
import torch
import torch.nn.functional as F
import numpy as np
import datetime
import time
from shutil import copyfile

from transformers import TransfoXLConfig, TransfoXLLMHeadModel, TransfoXLTokenizer
from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer
from transformers import BertForSequenceClassification, BertTokenizer

num_padded = 0
num_equal = 0
num_truncated = 0

# def split_data(datapath):
#     datapath = fixpath(datapath)
#     if not os.path.exists(datapath + 'ham') or not os.path.exists(datapath + 'spam'):
#         raise ValueError("Not a valid spam directory")
    
#     #plan is 80:20 train:test
#     # and then 80:20 within train for train:validate
#     # so get the test first, then split train
#     if not os.path.exists(datapath + 'test'):
#         os.mkdir(datapath + 'test')
#     if not os.path.exists(datapath + 'train'):
#         os.mkdir(datapath + 'train')
#     if not os.path.exists(datapath + 'dev'):
#         os.mkdir(datapath + 'dev')
    
#     spamdir = os.listdir(datapath + 'spam')
#     spamcount = len(spamdir)
#     hamdir = os.listdir(datapath + 'ham')
#     hamcount = len(hamdir)
#     for i in trange(int(spamcount/5), desc='test-spam'):
#         copyfile(datapath + 'spam/' + spamdir[i], datapath + 'test/' + spamdir[i]+'.spam')
#     for i in trange(int(hamcount/5), desc='test-ham '):
#         copyfile(datapath + 'ham/' + hamdir[i], datapath + 'test/' + hamdir[i]+'.ham')

#     #now get train and split 80:20 (ie every fifth goes to dev)
#     for i in tqdm(range(int(spamcount/5), spamcount), desc='train-spam'):
#         if i % 5 == 0:
#             copyfile(datapath + 'spam/' + spamdir[i], datapath + 'dev/' + spamdir[i]+'.spam')
#         else:
#             copyfile(datapath + 'spam/' + spamdir[i], datapath + 'train/' + spamdir[i]+'.spam')

#     for i in tqdm(range(int(hamcount/5), hamcount), desc='train-ham'):
#         if i % 5 == 0:
#             copyfile(datapath + 'ham/' + hamdir[i], datapath + 'dev/' + hamdir[i]+'.ham')
#         else:
#             copyfile(datapath + 'ham/' + hamdir[i], datapath + 'train/' + hamdir[i]+'.ham')

def get_first_occ(list, val):
    for i in range(len(list)):
        if list[i] == val:
            return i

    return -1

def fixpath(path):
    path_end = path[-1]
    if path_end != "\\" or path_end != "/":
        path += "/"

    return path

def getMessage(path, tokenizer, pad = False):

    #for diag
    global num_equal
    global num_padded
    global num_truncated

    with open(path, 'r') as msg:
        tokens = tokenizer.encode(msg.read(), add_special_tokens=True, max_length=512)
        if pad:
            count = len(tokens)
            if count < 512:
                num_padded += 1
                tokens += [-1] * (512 - count)
            elif count == 512:
                num_equal += 1
            else:
                num_truncated += 1
    
    #label
    if path.endswith('spam'):
        label = 1
    elif path.endswith('ham'):
        label = 0
    else:
        raise ValueError(f"Invalid class for file {path}")

    return tokens, label


def getBatch(datapath, batch_size, tokenizer):
    datapath = fixpath(datapath)
    count = 0
    batch_l = []
    files = os.listdir(datapath)
    for f in files:
        tokens, label = getMessage(datapath + f, tokenizer, batch_size != 1)
        batch_l += [tokens]
        count += 1
        if count % batch_size == 0:
            yield batch_l, label
            batch_l = []
        
            

def train(datapath, outpath, seed, batch_size, epochs, save_steps, use_cuda = True):
    #set up model and device (hopefully cuda)
    device = torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu")

    # if use_gpt:
    #     model = GPT2LMHeadModel.from_pretrained('gpt2')
    #     tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    # else:
    #     model = TransfoXLLMHeadModel.from_pretrained('transfo-xl-wt103')
    #     tokenizer = TransfoXLTokenizer.from_pretrained('transfo-xl-wt103')
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), betas=(.9,.999), eps=2e-05)
    
    #setup rng seeds on all devices to ensure repeatable results
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    num_batches = len(os.listdir(datapath)) / batch_size
    batch_list = getBatch(datapath, batch_size, tokenizer)

    avg_losses = []
    avg_loss = 0
    
    model.zero_grad()
    timestamp = datetime.datetime.now().strftime('%y%m%d%H%M%S')

    for _ in trange(epochs, desc="Epochs"):
        for batch_num in tqdm(range(0,int(num_batches), batch_size), desc="Batches"):
            #setup this batch.
            batch, label = next(batch_list)
            inputs = torch.tensor(batch, dtype=torch.long, device=device)
            labels = torch.tensor(label, dtype=torch.long, device=device)
            inputs = inputs.to(device)
            labels = labels.to(device)

            #feed input to model to train
            model.train()
            outputs = model(input_ids=inputs, labels=labels)

            # if not use_gpt:
            #     # loss returned from transfoXL was broken
            #     first_pad = get_first_occ(inputs[0], -1)
            #     loss = outputs[0][0][:first_pad].mean()

            loss = outputs[0]
            avg_loss += loss
            
            #update parameters
            loss.backward()
            optimizer.step()
            model.zero_grad()

            if batch_num % (batch_size * save_steps) == 0:
                print('CHECKPOINT')
                checkpoint_path = f"{fixpath(outpath)}{timestamp}/e{epochs}-num{batch_num}-size{batch_size}"
                if not os.path.exists(checkpoint_path):
                    os.makedirs(checkpoint_path)
                model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
                model_to_save.save_pretrained(checkpoint_path)
                tokenizer.save_pretrained(checkpoint_path)

                avg = avg_loss / save_steps
                print(f"average loss: {avg}")
                avg_losses += [avg]
                print('finished')
    
    print(avg_losses)

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--datapath", type=str, required=True,
    help="Input path of the training files (assuming using parsed trec datasets")

    parser.add_argument("--outpath", type=str, required=True,
    help="Where to put checkpoints for the model")

    parser.add_argument("--seed", type=int, default = 11122233, 
    help="RNG seed to make results reproducable")

    # parser.add_argument("--gpt2", action='store_true')

    parser.add_argument("--batch_size", type=int, required=True)

    parser.add_argument("--epochs", type=int, default=1)

    parser.add_argument("--no_cuda", action='store_true')

    parser.add_argument("--save_steps", type=int, default=250)

    args = parser.parse_args()

    # more arg checking. Make sure we can save models. Don't waste the training time
    if not os.path.exists(args.datapath):
        raise IOError ("Data path doesn't exist")

    if not os.path.exists(args.outpath):
        os.mkdir(args.outpath)

    # split_data(args.datapath)
    train(args.datapath, args.outpath, args.seed, args.batch_size, args.epochs, args.save_steps, not args.no_cuda)


if __name__ == "__main__":
    main()