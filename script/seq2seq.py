import cntk as C
from cntk.layers import *

import pickle
import numpy as np
import argparse

known, vocabs, chars = pickle.load(open('vocabs.pkl','rb'))

myConfig = {
        'save_name':'seq2seq',
        'save_freq': 10000,
        'output_dir':'v1',
        'max_epoch':70000,
        'epoch_size': 25000, # total: 44961 sequences in v1
        'batchsize':128,
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
    @C.Function
    def embed_layer(awk, awn):
        a_processed = C.times(awk, glove) + C.times(awn, nonglove)
        return a_processed
    return embed_layer

def create_model():
    '''
    return:model
    '''
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
        proj_out = Dense(myConfig['wg_dim']+myConfig['wn_dim'], input_rank=1)

        rec = Recurrence(rec_blocks[0])
        # last
        last_2_layer = Sequential([
            Recurrence(rec_blocks[1]),
            Recurrence(rec_blocks[2]),
            stab_out,
            proj_out,
            Label('out_proj_out')
            ])

        def decoder(history, inp):
            # encode
            encode_h, encode_c = encoder(inp).outputs
            # decode prepare
            his_stab = stab_in(history)
            # first layer
            out_fwd = ForwardDeclaration()
            attn = attention_model(encode_h, out_fwd)
            rec_in = C.splice(his_stab, attn)
            rec_out = rec(rec_in) # output
            rec_out_past = C.sequence.past_value(rec_out)
            out_fwd.resolve_to(rec_out_past)

            rec_stab = stab_out(rec_out)
            out_proj_out = last_2_layer(rec_stab)
            return out_proj_out

    return decoder

def create_train_model(s2smodel, embed_layer):
    '''
    return: @input map @softmax @loss
    '''
    q = C.Axis.new_unique_dynamic_axis('q')
    a = C.Axis.new_unique_dynamic_axis('a')
    b = C.Axis.default_batch_axis()
    qwk = C.sequence.input_variable(myConfig['wg_dim'], sequence_axis = q, is_sparse = False, name='qwk')
    qwn = C.sequence.input_variable(myConfig['wn_dim'], sequence_axis = q, is_sparse = False, name='qwn')
    awk = C.sequence.input_variable(myConfig['wg_dim'], sequence_axis = a, is_sparse = False, name='awk')
    awn = C.sequence.input_variable(myConfig['wn_dim'], sequence_axis = a, is_sparse = False, name='awn')

    input_ph = {'qwk':qwk,'qwn':qwn,'awk':awk,'awn':awn}
    #I = C.Constant(np.eye(myConfig['wg_dim']))
    #I2 = C.Constant(np.eye(myConfig['wn_dim']))

    #awkd, awnd = C.times(awk, I), C.times(awn, I2)
    #qwkd, qwnd = C.times(qwk, I), C.times(qwn, I2)
    a_processed = embed_layer(awk, awn)
    q_processed = embed_layer(qwk, qwn)
    q_onehot = C.splice(qwk, qwn)
    print("q_onehot shape:{}".format(q_onehot.output))

    logits = s2smodel(q_processed, a_processed)
    logits = C.sequence.slice(logits, 0, -1)
    print('logits shape:{}'.format(logits.output))

    labels = C.sequence.slice(q_onehot,1, 0) # <s> a b c </s> -> a b c </s>
    print('labels shape:{}'.format(labels.output))
    logits = C.reconcile_dynamic_axes(logits, labels)
    loss = C.cross_entropy_with_softmax(logits, labels)
    errs = C.classification_error(logits, labels)
    return input_ph, logits, C.combine(loss, errs)

def outer_process_history(embed_layer):
    @C.Function
    def process_history(hist, inp):
        wk = C.slice(hist, 0, 0, myConfig['wg_dim'])
        wn = hist[myConfig['wg_dim']:]
        hist_processed = embed_layer(wk, wn)
        out_logits = s2smodel(hist_processed, inp)
        hamax = C.hardmax(out_logits)
        return hamax
    return process_history

def create_eval_model(s2smodel, embed_layer, is_test=False):
    '''
    return: @input map @softmax @loss
    '''
    sentence_end_index = vocabs['</s>']
    q = C.Axis.new_unique_dynamic_axis('q')
    a = C.Axis.new_unique_dynamic_axis('a')
    b = C.Axis.default_batch_axis()
    qwk = C.sequence.input_variable(myConfig['wg_dim'], sequence_axis = q, is_sparse = False, name='qwk')
    qwn = C.sequence.input_variable(myConfig['wn_dim'], sequence_axis = q, is_sparse = False, name='qwn')
    awk = C.sequence.input_variable(myConfig['wg_dim'], sequence_axis = a, is_sparse = False, name='awk')
    awn = C.sequence.input_variable(myConfig['wn_dim'], sequence_axis = a, is_sparse = False, name='awn')

    input_ph = {'qwk':qwk,'qwn':qwn,'awk':awk,'awn':awn}

    @C.Function
    def greedy_model(aawk, aawn, qqwk, qqwn):
        q_onehot = C.splice(qqwk, qqwn)
        sentence_start = C.sequence.slice(q_onehot, 0, 1)

        @C.Function
        def process_history(hist, inp):
            wk = C.slice(hist, 0, 0, myConfig['wg_dim'])
            wn = hist[myConfig['wg_dim']:]
            hist_processed = embed_layer(wk, wn)
            out_logits = s2smodel(hist_processed, inp)
            hamax = C.hardmax(out_logits)
            return hamax

        a_processed = embed_layer(aawk, aawn)
        unfold = UnfoldFrom(lambda history: process_history(history, a_processed),
                until_predicate=lambda w:w[:sentence_end_index],
                length_increase=1.5)
        out_onehot = unfold(sentence_start, a_processed)
        return out_onehot

    #I = C.Constant(np.eye(myConfig['wg_dim']))
    #I2 = C.Constant(np.eye(myConfig['wn_dim']))

    #awkd, awnd = C.times(awk, I), C.times(awn, I)
    #qwkd, qwnd = C.times(qwk, I), C.times(qwn, I)
    q_onehot = C.splice(qwk, qwn)
    labels = C.sequence.slice(q_onehot,1, 0) # <s> a b c </s> -> a b c </s>

    out_onehot = greedy_model(awk, awn, qwk, qwn)
    out_onehot = C.sequence.unpack(out_onehot, 0, True) # no mask output
    out_onehot_trunc = C.to_sequence_like(out_onehot, labels)
    print('greedy_model onehot out:{}'.format(out_onehot_trunc.output))
    loss = C.cross_entropy_with_softmax(out_onehot_trunc, labels)
    errs = C.classification_error(out_onehot_trunc, labels)
    return input_ph, out_onehot, C.combine(loss, errs)

def create_reader(filename, input_ph, config, is_test=False):
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
            ),randomize=False if is_test else True, max_sweeps=5000 if is_test else C.io.INFINITELY_REPEAT)
    input_map = {
            input_ph['qwk']:mb_source.streams.qwk,
            input_ph['qwn']:mb_source.streams.qwn,
            input_ph['awk']:mb_source.streams.awk,
            input_ph['awn']:mb_source.streams.awn
            }
    return mb_source, input_map

def train(config, model, enable_eval=False):
    max_epoch = config['max_epoch']
    batchsize = config['batchsize']
    epoch_size = config['epoch_size']
    lr = config['lr']
    save_freq = config['save_freq']

    # create models
    embed_layer = GloveEmbed()

    inp_ph ,train_model, loss_errs = create_train_model(model, embed_layer)
    train_reader, input_map = create_reader('aq_train.ctf', inp_ph, config)

    if enable_eval:
        inp_ph2, greedy_model, loss_errs2 = create_eval_model(model, embed_layer)
        eval_reader, input_map2 = create_reader('aq_dev.ctf', inp_ph, config, True)
        evaluator = C.eval.Evaluator(loss_errs2.outputs[0])
        # i2w = get_i2w(vocabs)

    # create loggers
    progress_printer = C.logging.ProgressPrinter(freq=500, tag="Train")
    tensorboard_writer = C.logging.TensorBoardProgressWriter(500, 'tensorlog', model=train_model)

    lrs = [(1, lr), (5000, lr*0.1), (10000,lr*0.01), (30000, lr*0.001)]
    learner = C.fsadagrad(train_model.parameters,
         #apply the learning rate as if it is a minibatch of size 1
         lr = C.learning_parameter_schedule(lrs),
         momentum = C.momentum_schedule(0.9, minibatch_size=batchsize),
         gradient_clipping_threshold_per_sample=2,
         gradient_clipping_with_truncation=True)

    trainer = C.Trainer(train_model, loss_errs, [learner], [progress_printer, tensorboard_writer])

    total_samples = 0
    for epoch in range(max_epoch):
        while total_samples < (epoch+1) * epoch_size:
            # get next minibatch of training data
            mb_train = train_reader.next_minibatch(batchsize, input_map=input_map)
            # do the training
            trainer.train_minibatch(mb_train)
            total_samples += mb_train[list(mb_train.keys())[0]].num_sequences
            # total_samples += trainer.total_number_of_samples_seen

        trainer.summarize_training_progress()

        if epoch+1 % save_freq == 0:
            save_name = '{}_{}.model'.format(config['save_name'], epoch+1)
            print('save {} in {}'.format(save_name, config['output_dir']))
            trainer.save_checkpoint('output/{}/{}'.format(config['output_dir'], save_name))

            if enable_eval:
                # vis_mb = eval_reader.next_minibatch(1, input_map=input_map2)
                # oneh = greedy_model.eval(vis_mb)[0]
                # res = visualize(oneh, i2w)
                # print(res)
                while True:
                    mb_eval=eval_reader.next_minibatch(128, input_map=input_map2)
                    if not mb_eval:
                        break
                    evaluator.test_minibatch(mb_eval)
                evaluator.summarize_test_progress()

def evaluate(s2smodel, visual=True):
    pass

def get_i2w(vocab_dict):
    return {v:k for k,v in vocab_dic.items()}
def visualize(onehot, i2w):
    ''' @onehot:numpy matrix @i2w: worddict'''
    idx = [np.argmax(oo) for oo in onehot]
    return [i2w[i] for i in idx]

C.cntk_py.set_gpumemory_allocation_trace_level(0)
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu',help='specify gpu id', default=0, type=int)

    args = parser.parse_args()
    C.try_set_default_device(C.gpu(args.gpu))

    s2smodel = create_model()
    train(myConfig ,s2smodel, True)
