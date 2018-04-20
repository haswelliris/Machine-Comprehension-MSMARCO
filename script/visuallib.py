import pickle, argparse, os
import tsv2ctf
import numpy as np
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('pickle_file', help='pickle weight file to visualize')
    parser.add_argument('ref_file', help='reference file')
    args = parser.parse_args()

    # imports required for showing the attention weight heatmap
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    with open(args.file, 'rb') as f:
        weights = pickle.load(f)
        with open(args.ref_file, encoding='utf8') as ref:
            ll = len(weights[0])
            rows = [ref.readline() for _ in range(ll)]

        contents = [l.strip().split('\t')[2:4] for l in rows] # c,q
        for mm in weights: # different weights
            for i,ww in enumerate(mm): # different sequence
                ww = np.squeeze(np.array(ww))
                columns = contents[i][0].split()
                if len(ww)==ww.shape[1]:
                    index = columns
                else:
                    index = contents[i][1].split()
                dframe = pd.DataFrame(data=ww.T, columns=columns, index=index)
                sns.heatmap(dframe)
                plt.show()
