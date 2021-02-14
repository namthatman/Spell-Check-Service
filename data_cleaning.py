# Clean Raw Articles CSV

# Library

# Data Processing
import numpy as np
import pandas as pd
import re
from fairseq.data import Dictionary


# Load dataset
#raw_articles = pd.read_csv('resources/train_dataset/dim_articles.csv', sep='\t')
input_articles = pd.read_csv("resources/train_dataset/false_input_articles_full.csv", header=None)
output_articles = pd.read_csv("resources/train_dataset/false_output_articles_full.csv", header=None)


# Reduce into one dimension
#dim_articles = []

def clean_paragraphs(text):
    text = text.replace('","', ' ')
    text = re.sub(r'"', "", text)
    text = re.sub(r"[]'[]", "", text)
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"\\u.{4}", "", text)
    text = re.sub(r"\\", "", text)
    text = re.sub(r"  ", " ", text)
    return text


#text1 = raw_articles['paragraphs'][2]
#cleaned_text = clean_paragraphs(text1).split(". ")[:-1]
#for title in raw_articles['title'].values:
#    dim_articles.append(clean_paragraphs(title))
    
#for para in raw_articles['paragraphs'].values:
#    list_text = clean_paragraphs(para).split(". ")[:-1]
#    dim_articles.extend(list_text)
    

def extract_corpus(para):
    list_text = clean_paragraphs(para).split(". ")
    return list_text


# Save dataframe into csv
#articles_df = pd.DataFrame(dim_articles)
#articles_df.to_csv('dim_articles.csv', index=False, header=False)


# Load new dataset
#new_articles = pd.read_csv('dim_articles.csv', header=None)
    

def encode_input(test_input, MAX_LEN, vocab):
    cs = len(test_input)
    input_ids_s = np.ones((cs,MAX_LEN),dtype='int32')
    attention_mask_s = np.zeros((cs,MAX_LEN),dtype='int32')
    token_type_ids_s = np.zeros((cs,MAX_LEN),dtype='int32')
    
    for k in range(cs):
        text = "<s> " + str(test_input[k]) + " </s>"
        enc = vocab.encode_line(text, append_eos=False, add_if_not_exist=False).long().tolist()
        input_ids_s[k,:len(enc)] = enc
        attention_mask_s[k,:len(enc)] = 1
        
    return input_ids_s, attention_mask_s, token_type_ids_s


def get_max_len(df):
    length = []
    for text in df:
        length.append(len(str(text).split(" ")))
    return [min(length), max(length), np.mean(length), np.median(length), np.std(length)]

#result = get_max_len(input_articles.values)




