import re
import time

import numpy as np
import pandas as pd
from datasketch import MinHash, MinHashLSH, MinHashLSHForest

lines = []
with open("../data/ftp.mokk.bme.hu/Hunglish2/modern.lit/bi/hunglish1.lit.bi", 'r', encoding='utf_8') as file:
    for l in file:
        lines.append(re.split(r'\t+', l.strip()))

db = pd.DataFrame(lines, columns=["hun", "en"])
permutations = 128
num_recommendations = 2

def preprocess(text):
    text = re.sub(r'[^\w\s]','',text)
    tokens = text.lower()
    tokens = tokens.split()
    return tokens

def get_forest(data, perms):
    minhash = []
    
    for text in data["hun"]:
        tokens = preprocess(text)
        m = MinHash(num_perm=perms)
        for s in tokens:
            m.update(s.encode('utf8'))
        minhash.append(m)
        
    lsh = MinHashLSH(threshold=0.7, num_perm=perms)
    for i,m in enumerate(minhash):
        lsh.insert(i,m)
    
    return lsh

def predict(text, database, perms, num_results, forest):
    start_time = time.time()
    
    tokens = preprocess(text)
    m = MinHash(num_perm=perms)
    for s in tokens:
        m.update(s.encode('utf8'))
        
    idx_array = np.array(forest.query(m))

    if len(idx_array) == 0:
        return None
    
    return idx_array

def make_sets(l):
    ret = []
    for i in range(len(l)):
        for j in range(i+1, len(l)):
            ret.append((l[i],l[j]))
    return ret

forest = get_forest(db, permutations)
pairs = set()
for idx in range(len(db)):
    pred = predict(db["hun"][idx], db, permutations, num_recommendations, forest)
    sets = make_sets(sorted(pred))
    for s in sets:
        pairs.add(s)
print(pairs)