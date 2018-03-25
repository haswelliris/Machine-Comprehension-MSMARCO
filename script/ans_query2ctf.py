import pickle
import numpy as np
from config import *
from tsv2ctf import populate_dicts
from itertools import zip_longest

word_count_threshold = data_config['word_count_threshold']
char_count_threshold = data_config['char_count_threshold']
word_size = data_config['word_size']

sanitize = str.maketrans({"|": None, "\n": None})
tsvs = ('train', 'dev')
unk = '<UNK>'
pad = ''
EMPTY_TOKEN = '<NULL>'
START_TOKEN = '<s>'
END_TOKEN = '</s>'
# pad (or trim) to word_size characters
pad_spec = '{0:<%d.%d}' % (word_size, word_size)

def tsv_iter(line, vocab):
    unk_w = vocab[unk]

    uid, title, context, query, answer, raw_context, begin_answer, end_answer, raw_answer = line.split('\t')
    atokens = raw_answer.split(' ')
    qtokens = query.split(' ')

    atokens = [START_TOKEN]+[t if t!= EMPTY_TOKEN else '' for t in atokens]+[END_TOKEN]
    qtokens = [START_TOKEN]+[t if t!=EMPTY_TOKEN else '' for t in qtokens]+[END_TOKEN]

    awids = [vocab.get(t.lower(),unk_w) for t in atokens]
    qwids = [vocab.get(t.lower(), unk_w) for t in qtokens]
    
    return atokens, qtokens, awids, qwids

def tsv_to_ctf(f, g, vocab, known):
    st_w = vocab[START_TOKEN]
    ed_w = vocab[END_TOKEN]

    for i,line in enumerate(f):
        atokens, qtokens, awids, qwids = tsv_iter(line, vocab)
        for atoken, qtoken, awid, qwid in zip_longest(atokens, qtokens, awids, qwids):
            out = [str(i)]
            if atoken is not None:
                out.append('|# %s'%pad_spec.format(atoken.translate(sanitize)))
            if qtoken is not None:
                out.append('|# %s'%pad_spec.format(qtoken.translate(sanitize)))
            if awid is not None:
                if awid<known:
                    out.append('|awk {}:{}'.format(awid,1))
                    out.append('|awn {}:{}'.format(0,0))
                else:
                    out.append('|awk {}:{}'.format(0,0))
                    out.append('|awn {}:{}'.format(awid-known,1))
            if qwid is not None:
                if qwid<known:
                    out.append('|qwk {}:{}'.format(qwid,1))
                    out.append('|qwn {}:{}'.format(0,0))
                else:
                    out.append('|qwk {}:{}'.format(0,0))
                    out.append('|qwn {}:{}'.format(qwid-known,1))
            g.write('\t'.join(out))
            g.write('\n')
                
    
# discard unknown words
if __name__=='__main__':
    try:
        known, vocab, chars = pickle.load(open('vocabs.pkl','rb'))
    except:
        known, vocab, chars = populate_dicts(tsvs)
        f = open('vocabs.pkl','wb')
        pickle.dump((known,vocab,chars), f)
        f.close()
    for tsv in tsvs:
        with open('%s.tsv'%tsv,'r',encoding='utf8') as f:
            with open('%s.ctf'%('aq_'+tsv),'w', encoding='utf8') as g:
                tsv_to_ctf(f, g, vocab,known) 
