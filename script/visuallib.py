import pickle, argparse, os
import tsv2ctf
import numpy as np
# imports required for showing the attention weight heatmap
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
def preload_vobs(pklname):
    with open(pklname,'rb') as f:
        _, vocabs, _ = pickle.load(f)
    i2w = {v:k for k, v in vocabs.items()}
    return i2w, vocabs
def visualize(pklname, i2w):
    with open(pklname, 'rb') as f:
        # [array(len)] allWei:[(embed, attn)...]
        allQid, allDid, allWei = pickle.load(f)
    
    for qids, dids, mm in zip(allQid, allDid, allWei): # different weights
        columns = [i2w.get(wid, '<UNK>') for wid in qids]
        index = [i2w.get(wid,'<UNK>') for wid in dids] 
        for emb,ww in zip(mm): # different sequence
            if len(columns)==ww.shape[1]:
                dframe = pd.DataFrame(data=ww, columns=columns, index=index)
            elif len(index)==ww.shape[1]:
                dframe = pd.DataFrame(data=ww, columns=index, index=columns)
            elif len(columns)==emb.shape[1]:
                dframe = pd.DataFrame(data=emb, columns=columns, index=index)
            elif len(index)==embed.shape[1]:
                dframe = pd.DataFrame(data=ww, columns=index, index=columns)
            else:
                continue
            sns.heatmap(dframe)
            plt.show()
    