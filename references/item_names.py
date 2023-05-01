import os
import re
import glob
import pandas as pd
import numpy as np
from difflib import SequenceMatcher

# read file
with open("item-names.csv", "r") as f:
    data = [re.sub(r"�+", "'", l).strip().split(";", -1) for l in f.readlines()]


# make pandas dataframe
questions = []
keys = []
for d in data:

    questions.append(' '.join(d[:-1]))
    if len(d)>1:
        keys.append(d[-1])
    else:
        keys.append(None)

df = pd.DataFrame(np.array([questions, keys]).T, columns=['questions', 'keys'])

# get all data dictionaries
os.chdir('Release9_DataDic')
data_dics = glob.glob('*xlsx')

dicts = {}
for data_dic in data_dics:
    if '~$' not in data_dic:
        df_dict = pd.read_excel(data_dic)
        dict_list = df_dict.to_numpy().flatten()
        key = data_dic.strip('.xlsx')
        dicts.update({key: dict_list})

keys_mat = np.array(np.zeros((len(df),2)), dtype=object)
for idx in df.index:
    datadics = []
    # loop over dictionary keys
    for k,v in dicts.items():
        if df.loc[idx, 'keys'] in v:
            datadics.append(k)
    if 0<len(datadics)<=2:
        keys_mat[idx,:] = datadics
    else:
        keys_mat[idx,:] = 'No Key'

# now loop over `keys` and link keys to data dictionary
for idx in df.index:

    similarity = SequenceMatcher(None, keys_mat[idx,0], keys_mat[idx,1]).ratio()

    # if keys have two corresponding measures
    # check which sentences match and return most likely measure
    if similarity==1:
        df.loc[idx,'datadic'] = keys_mat[idx][0]
    else:
        removelist = " "
        question = re.sub(r'[^\w'+removelist+']', '', df.loc[idx, 'questions'])
        ratios = {}
        for kk in keys_mat[idx]:
            for vv in dicts[kk]:
                print(f'comparing {question} to {vv}')
                try:
                    # strip non-alphanumeric characters from strings and compare
                    compare_str = re.sub(r'[^\w'+removelist+']', '', vv)
                    ratio = SequenceMatcher(None, question, compare_str).ratio()
                    ratios.update({f'{kk}:{compare_str}': ratio})
                except:
                    pass
        # find matching sentence
        sentence_idx = np.argmax(list(ratios.values()))
        correct_key, correct_sentence = list(ratios.keys())[sentence_idx].split(':')
        #df.loc[idx, 'new_question'] = correct_sentence
        df.loc[idx, 'datadic'] = correct_key


# save out new file
os.chdir('../')
df.to_csv('item-names-new.csv', index=False)