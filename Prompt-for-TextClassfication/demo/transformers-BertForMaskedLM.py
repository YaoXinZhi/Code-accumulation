# -*- coding:utf-8 -*-
# ! usr/bin/env python3
"""
Created on 23/08/2021 14:57
@Author: yao
"""

from transformers import BertTokenizer, BertForMaskedLM
import torch

# tokenizer.vocab_size = 30522
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForMaskedLM.from_pretrained('bert-base-uncased')

inputs = tokenizer("The capital of France is [MASK].", return_tensors="pt")
labels = tokenizer("The capital of France is Paris.", return_tensors="pt")["input_ids"]

# outputs.keys -- loss, logits(1,9,30522)
outputs = model(**inputs, labels=labels)
loss = outputs.loss
logits = outputs.logits


