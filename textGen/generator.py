#!/usr/bin/env python3
from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import logging
import os
from BERTTrainer import fixpath
from tqdm import trange, tqdm

import torch
import torch.nn.functional as F
import numpy as np

from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer


# logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
#                     datefmt = '%m/%d/%Y %H:%M:%S',
#                     level = logging.INFO)
# logger = logging.getLogger(__name__)

def generate(args):    
    # check for cuda first.
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")

    if args.modelpath == "":
        args.modelpath = "gpt2"

    failures = 0

    model = GPT2LMHeadModel.from_pretrained(args.modelpath)
    tokenizer = GPT2Tokenizer.from_pretrained(args.modelpath)
    model.to(device)

    for cur_seed in trange(args.seed, args.seed + args.samples):
        np.random.seed(cur_seed)
        torch.manual_seed(cur_seed)
        torch.cuda.manual_seed_all(cur_seed)

        tokens = tokenizer.encode(args.prompt, add_special_tokens=False)
        text_tensor = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)

        with torch.no_grad():
            model.eval() # Evaluating, not training. Don't edit weights.
            for _ in range(args.length):
                inputs = {'input_ids': text_tensor} #copying example code because it works
                outputs = model(**inputs)
                next_token_logits = outputs[0][:,-1,:]
                # pick one token from logit scores.
                next_token = torch.multinomial(F.softmax(next_token_logits, dim=-1), num_samples=1)
                text_tensor = torch.cat((text_tensor, next_token), dim=1)

        if args.prompt != " ":
            label = "spam"
        else:
            label = "ham"

        try:
            with open(fixpath(args.outpath) + f'{cur_seed}.{label}', 'w+') as f:
                f.write(tokenizer.decode(text_tensor.tolist()[0]))
        except UnicodeEncodeError:
            failures += 1
            continue
        # for token in text_tensor[:,:].tolist():
        #     out = tokenizer.decode(token, clean_up_tokenization_spaces = True)

        #     print(out)
    print(failures)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--prompt", type=str, default=" ")
    parser.add_argument("--seed", type=int, default=11122233)
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--length", type=int, default=512)
    parser.add_argument("--modelpath", type=str, default="")
    parser.add_argument("--no_cuda", action='store_true')
    parser.add_argument("--outpath", type=str, required=True)

    args = parser.parse_args()

    if not os.path.exists(args.outpath):
        os.mkdir(args.outpath)
    generate(args)