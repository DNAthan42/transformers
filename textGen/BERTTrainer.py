import argparse
import os
from tqdm import trange, tqdm
import torch
import torch.nn.functional as F
import numpy as np
import datetime
import time

# from transformers import TransfoXLConfig, TransfoXLLMHeadModel, TransfoXLTokenizer
# from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer
from transformers import BertForSequenceClassification, BertTokenizer

num_padded = 0
num_equal = 0
num_truncated = 0

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
        tokens = tokenizer.encode(msg.read(), add_special_tokens=False, max_length=512)
        if pad:
            count = len(tokens)
            if count < 512:
                num_padded += 1
                tokens += [-1] * (512 - count)
            elif count == 512:
                num_equal += 1
            else:
                num_truncated += 1
    
    return tokens


def getBatch(datapath, batch_size, tokenizer):
    datapath = fixpath(datapath)
    count = 0
    batch_l = []
    files = os.listdir(datapath)
    for f in files:
        batch_l += [getMessage(datapath + f, tokenizer, batch_size != 1)]
        count += 1
        if count % batch_size == 0:
            yield batch_l
            batch_l = []
        
            

def train(datapath, outpath, seed, batch_size, epochs, save_steps, use_gpt, use_cuda = True):
    #set up model and device (hopefully cuda)
    device = torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu")

    if use_gpt:
        model = GPT2LMHeadModel.from_pretrained('gpt2')
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    else:
        model = TransfoXLLMHeadModel.from_pretrained('transfo-xl-wt103')
        tokenizer = TransfoXLTokenizer.from_pretrained('transfo-xl-wt103')
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), betas=(.9,.98), eps=1e-09)
    
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
            batch = torch.tensor(next(batch_list), dtype=torch.long, device=device)
            inputs, labels = batch, batch
            inputs = inputs.to(device)
            labels = labels.to(device)

            #feed input to model to train
            model.train()
            outputs = model(input_ids=inputs, labels=labels)

            if use_gpt:
                # loss returned from transfoXL was broken
                first_pad = get_first_occ(inputs[0], -1)
                loss = outputs[0][0][:first_pad].mean()

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

    parser.add_argument("--gpt2", action='store_true')

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

    train(args.datapath, args.outpath, args.seed, args.batch_size, args.epochs, args.save_steps, args.gpt2, not args.no_cuda)


if __name__ == "__main__":
    main()