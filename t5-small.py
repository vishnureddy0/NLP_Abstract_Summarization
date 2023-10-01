# -*- coding: utf-8 -*-
"""
Created on Fri Oct  8 00:21:09 2021

@author: akarsh
"""

import torch
import json 
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config
import pandas as pd

model = T5ForConditionalGeneration.from_pretrained('t5-small')
tokenizer = T5Tokenizer.from_pretrained('t5-small')
device = torch.device('cpu')

df = pd.read_excel(r"E:\zee_proj\exp\code\news_data.xlsx",sheet_name = 0)

summaries=[]           
             
for ind in range(3,10):
    text = df['Short'][ind]
    
    
    preprocess_text = text.strip().replace("\n","")
    t5_prepared_Text = "summarize: "+preprocess_text
    # print ("original text preprocessed: \n", preprocess_text)
    
    tokenized_text = tokenizer.encode(t5_prepared_Text, return_tensors="pt").to(device)
    
    
    # summmarize 
    summary_ids = model.generate(tokenized_text,
                                        num_beams=4,
                                        no_repeat_ngram_size=2,
                                        min_length=30,
                                        max_length=100,
                                        early_stopping=True)
    
    output = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    
    summaries.append(output)