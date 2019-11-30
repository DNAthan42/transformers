#!/usr/bin/env python3
from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import logging
from tqdm import trange

import torch
import torch.nn.functional as F
import numpy as np

from transformers import TransfoXLLMHeadModel, TransfoXLTokenizer, TransfoXLConfig


logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

def generate(prompt, seed = 11122233, length = 150, use_cuda = True):    
    # check for cuda first.
    device = torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu")

    model = TransfoXLLMHeadModel.from_pretrained('transfo-xl-wt103')
    tokenizer = TransfoXLTokenizer.from_pretrained('transfo-xl-wt103')
    model.to(device)
    model.eval() # Evaluating, not training. Don't edit weights.

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    tokens = tokenizer.encode(prompt, add_special_tokens=False)
    text_tensor = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)

    with torch.no_grad():
        for _ in trange(length):
            inputs = {'input_ids': text_tensor}
            outputs = model(**inputs)
            next_token_logits = outputs[0][:,-1,:]
            next_token = torch.multinomial(F.softmax(next_token_logits, dim=-1), num_samples=1)
            text_tensor = torch.cat((text_tensor, next_token), dim=1)

    for token in text_tensor[:,:].tolist():
        out = tokenizer.decode(token, clean_up_tokenization_spaces = True)

        print(out)

    return out

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--seed", type=int, default=11122233)

    args = parser.parse_args()
    generate(args.prompt)