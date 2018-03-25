import cntk as C
from cntk.layers import *

import pickle
import numpy as np

known, vocabs, chars = pickle.load(open('vocabs.pkl','rb'))

myConfig = {
        'max_epoch':70000,
        'batchsize':256,
        'lr':0.1,
        'wg_dim':known,
        'wn_dim':len(vocabs)-known,
        'embed_dim':300,
        'hidden_dim':150,
        'attention_dim':150,
        'layers':3
        }

# ============ layers ==============
def GloveEmbed():
    def embed_layer(awk, awn):
        # load parameters
        npglove = np.zeros((myConfig['wg_dim'], myConfig['embed_dim']), dtype=np.float32)
        with open('glove.6B.300d.txt',encoding='utf8') as f:
            for line in f:
                parts = line.split()
                word = parts[0].lower()
                idx = vocabs[word]
                if idx<known:
                    npglove[idx] = np.array([float(p) for p in parts[1:]])
        glove = C.constant(npglove)
        nonglove = C.parameter(shape=(myConfig['wn_dim'], myConfig['embed_dim']), init = C.glorot_uniform(), name='nongloveE')

        a_processed = C.times(awk, glove) + C.times(awn, nonglove)

        return a_processed
    return embed_layer


def create_model(is_train=True):
    '''
    return:model, placeholder
    '''
    q = C.Axis.new_unique_dynamic_axis('q')
    a = C.Axis.new_unique_dynamic_axis('a')
    b = C.Axis.default_batch_axis()
    qwk = C.sequence.input_variable(myConfig['wg_dim'], sequence_axis = q, is_sparse = False, name='qwk')
    qwn = C.sequence.input_variable(myConfig['wn_dim'], sequence_axis = q, is_sparse = False, name='qwn')
    awk = C.sequence.input_variable(myConfig['wg_dim'], sequence_axis = a, is_sparse = False, name='awk')
    awn = C.sequence.input_variable(myConfig['wn_dim'], sequence_axis = a, is_sparse = False, name='awn')

    input_ph = {'qwk':qwk,'qwn':qwn,'awk':awk,'awn':awn}
    embed_layer = GloveEmbed()

    a_processes = embed_layer(awk, awn)

    with default_options(enable_self_stabilization=True):
        encoder = Sequential([
            Stabilizer(),
            For(range(myConfig['layers']-1), lambda:Recurrence(LSTM(myConfig['hidden_dim']))),
            Recurrence(LSTM(myConfig['hidden_dim']),return_full_state=True),
            (Label('encode_h'),Label('encode_c'))
            ])
    with default_options(enable_self_stabilization=True):
        stab_in = Stabilizer()
        stab_out = Stabilizer()
        attention_model = AttentionModel(myConfig['attention_dim'])
        rec_blocks = [LSTM(myConfig['hidden_dim']) for _ in range(myConfig['layers'])]
        proj_out = Dense(myConfig['wg_dim']+myConfig['wn_dim'])

        rec = Recurrence(rec_blocks[0])
        # last
        last_2_layer = Sequential([
            Recurrence(rec_blocks[1]),
            Recurrence(rec_blocks[2]),
            stab_out,
            proj_out,
            Label('out_proj_out')
            ])

        def decoder(his, inp):
            # first layer
            attn = attention_model(encode_h, out_fwd)
            rec_in = C.splice(a_stab, attn)
            rec_out = rec(rec_in) # output
            rec_out_past = C.sequence.past_value(rec_out)
            out_fwd.resolve_to(rec_out_past)
            out_proj_out = last_2_layer(rec_out)
            return out_proj_out

    return decoder, input_ph

def create_train_model(s2smodel):
    def train_model(inp_wk, inp_wn, label_wk, label_wn):
        past_label =

def create_reader(filename, input_ph, config):
    '''
    return CTFReader
    '''
    mb_source = C.io.MinibatchSource(
            C.io.CTFDeserializer(
                filename,
                C.io.StreamDefs(
                    qwk = C.io.StreamDef('qwk',shape=config['wg_dim'], is_sparse=True),
                    qwn = C.io.StreamDef('qwn', shape=config['wn_dim'], is_sparse=True),
                    awk = C.io.StreamDef('awk', shape=config['wg_dim'], is_sparse=True),
                    awn = C.io.StreamDef('awn', shape=config['wn_dim'], is_sparse=True)
                )
            ),randomize=True, max_sweeps=C.io.INFINITELY_REPEAT)
    input_map = {
            input_ph['qwk']:mb_source.streams.qwk,
            input_ph['qwn']:mb_source.streams.qwn,
            input_ph['awk']:mb_source.streams.awk,
            input_ph['awn']:mb_source.streams.awn
            }
    return mb_source, input_map

def train(config, model):
    max_epoch = config['max_epoch']
    batchsize = config['batchsize']
    lr = config['lr']



def evaluate(model):
    pass

if __name__=='__main__':
    s2smodel,input_ph = create_model()
    train_reader, input_map = create_reader('aq_train.ctf' ,input_ph, myConfig)
    batch = train_reader.next_minibatch(myConfig['batchsize'], input_map)
    print(batch)
