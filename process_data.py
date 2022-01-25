#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import json
import re
from collections import Counter
import numpy as np
from ast import literal_eval
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer


# ### Prepare Dataset

# In[2]:


def preprocess_method(files):
    linked_methods = []
    files = json.loads(files.replace('\'', '"'))
    for file in files:
        normalized_method_name = re.sub('([a-zA-Z]+ )', r' \1', ' '.join(file['methods']))
        linked_methods.append(str.lower(normalized_method_name))
    return re.sub(r'[^a-zA-Z]', ' ', ' '.join(linked_methods))

def preprocess_bug_reports(text):
    normalized = re.sub('([a-zA_Z]+ )', r' \1', str.lower(text)).split()
    return re.sub(r'[^a-z]', ' ', ' '.join(normalized))

def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    word_tokens = set(word_tokenize(text))
    filtered_sentence = [w for w in word_tokens if not w.lower() in stop_words]
    return ' '.join(filtered_sentence)

def stem_text(text):
    ps = PorterStemmer()
    res = []
    for t in text.split():
        res.append(ps.stem(t))
    return ' '.join(set(res))


# In[78]:


def preprocess_dataset(csv_path):
    df = pd.read_csv(file_path)
    df = df[['id', 'summary', 'description', 'files']]
    df['methods_raw'] = df['files'].apply(lambda x: preprocess_method(x))
    df['br_text_raw'] = df.apply(lambda x: preprocess_bug_reports(f"{x['summary']} {x['description']}"), axis=1)
    df['methods_sw1'] = df['methods_raw'].apply(lambda x: remove_stopwords(x))
    df['br_text_sw1'] = df['br_text_raw'].apply(lambda x: remove_stopwords(x))
    df['methods_stemmed'] = df['methods_sw1'].apply(lambda x: stem_text(x))
    df['br_text_stemmed'] = df['br_text_sw1'].apply(lambda x: stem_text(x))
    df['methods_sw2'] = df['methods_stemmed'].apply(lambda x: remove_stopwords(x))
    df['br_text_sw2'] = df['br_text_stemmed'].apply(lambda x: remove_stopwords(x))
    return df

def extract_tokens(df):
    raw_methods_tokens = ' '.join(df['methods_raw'])
    method_tokens = Counter(raw_methods_tokens.split())
    raw_br_tokens = ' '.join(df['br_text_raw'])
    br_tokens = Counter(raw_br_tokens.split())
    sw1_methods_tokens = ' '.join(df['methods_sw1'])
    sw1_methods = Counter(sw1_methods_tokens.split())
    sw1_br_tokens = ' '.join(df['br_text_sw1'])
    sw1_br = Counter(sw1_br_tokens.split())
    stemmed_methods_tokens = ' '.join(df['methods_stemmed'])
    stemmed_methods = Counter(stemmed_methods_tokens.split())
    stemmed_br_tokens = ' '.join(df['br_text_stemmed'])
    stemmed_br = Counter(stemmed_br_tokens.split())
    sw2_methods_tokens = ' '.join(df['methods_sw2'])
    sw2_methods = Counter(sw2_methods_tokens.split())
    sw2_br_tokens = ' '.join(df['br_text_sw2'])
    sw2_br = Counter(sw2_br_tokens.split())
    
    def print_pre_and_post_processing_report():
        print("### BR Tokens ###")
        print(f"### raw BR tokens (before preprocessing): {len(br_tokens)}")
        print(f"### sw1 BR tokens (sw removal): {len(sw1_br)}")
        print(f"### stemmed BR tokens (sw+stemming): {len(stemmed_br)}")
        print(f"### sw2 BR tokens (sw+stemming+sw): {len(sw2_br)}")
        print(f"-------------------------------------------------------------------")
        print("### SC Tokens ###")
        print(f"### raw SC tokens (before preprocessing): {len(method_tokens)}")
        print(f"### sw1 SC tokens (sw removal): {len(sw1_methods)}")
        print(f"### stemmed SC tokens (sw+stemming): {len(stemmed_methods)}")
        print(f"### sw2 SC tokens (sw+stemming+sw): {len(sw2_methods)}")
    print_pre_and_post_processing_report()
    return sw2_methods, sw2_br

def remove_outlier(sc_tokens, br_tokens, q=[25, 75], min_length=3):
    sc_threshold = np.percentile(list(sc_tokens.values()), q)
    print(f"sc_threshold: {sc_threshold}")
    br_threshold = np.percentile(list(br_tokens.values()), q)
    print(f"br_threshold: {br_threshold}")
    final_br = {}
    for s in br_tokens:
        if br_tokens[s] >= br_threshold[0] and br_tokens[s] <= br_threshold[1] and len(s) > min_length:
            final_br[s] = br_tokens[s]

    final_sc = {}
    for s in sc_tokens:
        if sc_tokens[s] >= sc_threshold[0] and sc_tokens[s] <= sc_threshold[1] and len(s) > min_length:
            final_sc[s] = sc_tokens[s]

    print("### BR Tokens ###")
    print(f"### Before removing outlier: {len(br_tokens)}")
    print(f"### After removing outlier {q}: {len(final_br)}")
    print(f"-------------------------------------------------------------------")
    print("### SC Tokens ###")
    print(f"### Before removing outlier: {len(sc_tokens)}")
    print(f"### After removing outlier {q}: {len(final_sc)}")
    
    return final_sc, final_br

def write_vocab_to_file(output_path, type, data):
    with open(f'{output_path}/{type}_occurence.json', mode='w') as f:
        f.write(json.dumps(dict(Counter(data).most_common())))
        
    vocabulary = {}
    idx = 0
    for m in data:
        vocabulary[m] = idx
        idx+=1
    with open(f'{output_path}/{type}_dictionary.json', mode='w') as f:
        f.write(json.dumps(dict(vocabulary)))
        
def filter_tokens(text, vocab):
    tokens = text.split()
    res = []
    for token in tokens:
        if token in vocab:
            res.append(token)
    return ' '.join(res)
        
def build_compact_matrix(df, br_vocab_id, sc_vocab_id, br_tokens, sc_tokens):
    df = df[['id', 'methods_sw2', 'br_text_sw2']].rename(columns={'methods_sw2': 'sc_tokens',
                                                                 'br_text_sw2': 'br_tokens'})
    df['filtered_br'] = df['br_tokens'].apply(lambda x: filter_tokens(x, br_tokens))
    df['filtered_sc'] = df['sc_tokens'].apply(lambda x: filter_tokens(x, sc_tokens))
    matrix = {} # matrix[<br_vocab_id>]={[<sc_tokens_id>=5.0]}
    for idx, row in df.iterrows():
        for br_token in row['filtered_br'].split():
            br_token_id = br_vocab_id[br_token]
            matrix[br_token_id] = matrix.get(br_token_id, {})
            for sc_token in row['filtered_sc'].split():
                sc_token_id = sc_vocab_id[sc_token]
                matrix[br_token_id][sc_token_id] = 5       
    return matrix

def generate_2d_matrix(matrix, br_vocab_id, sc_vocab_id):
    token_matrix = np.zeros((len(br_vocab_id), len(sc_vocab_id)), dtype=int)
    for br_id in matrix:
        for sc_id in matrix[br_id]:
            token_matrix[int(br_id)][int(sc_id)] = 5.0
            
def generate_training_file(matrix, br_token, sc_token):
    temp = []
    for br_token_id in matrix:
        for sc_token_id in range(len(sc_token)):
            if matrix[br_token_id].get(sc_token_id):
                temp.append((br_token_id, sc_token_id, matrix[br_token_id].get(sc_token_id,0)))
    return pd.DataFrame(temp, columns=['br_token', 'sc_token', 'score'])
            
file_path = '/Users/ivanaclairineirsan/Documents/Research/Bug Localization/IMSE2021/gen_csv_raw/Apache/CAMEL.csv'
output_path = '/Users/ivanaclairineirsan/Documents/Research/QT4BL/data/Bench4BL/Apache/CAMEL'
df = preprocess_dataset(file_path)
sc_tokens_bef, br_tokens_bef = extract_tokens(df)
sc_tokens, br_tokens = remove_outlier(sc_tokens_bef, br_tokens_bef)
write_vocab_to_file('/Users/ivanaclairineirsan/Documents/Research/QT4BL/data/Bench4BL/Apache/CAMEL', 'br', br_tokens)
write_vocab_to_file('/Users/ivanaclairineirsan/Documents/Research/QT4BL/data/Bench4BL/Apache/CAMEL', 'sc', sc_tokens)

with open(f'{output_path}/sc_dictionary.json', 'r') as f:
    sc_vocab_id = json.load(f)
    
with open(f'{output_path}/br_dictionary.json', 'r') as f:
    br_vocab_id = json.load(f)

matrix = build_compact_matrix(df, br_vocab_id, sc_vocab_id, br_tokens, sc_tokens)
train_df = generate_training_file(matrix,br_vocab_id,sc_vocab_id)
train_df.to_csv('train_data.csv')