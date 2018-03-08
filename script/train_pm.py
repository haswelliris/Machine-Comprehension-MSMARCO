# -*- coding:utf8
import cntk as C
import numpy as np
from polymath import PolyMath
from squad_utils import metric_max_over_ground_truths, f1_score, exact_match_score
import tsv2ctf
import os
import argparse
import importlib
import time
import json

model_name = "pm.model"
log_file_name = 'default_output'

def argument_by_name(func, name):
    found = [arg for arg in func.arguments if arg.name == name]
    if len(found) == 0:
        raise ValueError('no matching names in arguments')
    elif len(found) > 1:
        raise ValueError('multiple matching names in arguments')
    else:
        return found[0]

def create_mb_and_map(func, data_file, polymath, randomize=True, repeat=True):
    mb_source = C.io.MinibatchSource(
        C.io.CTFDeserializer(
            data_file,
            C.io.StreamDefs(
                context_g_words  = C.io.StreamDef('cgw', shape=polymath.wg_dim,     is_sparse=True),
                query_g_words    = C.io.StreamDef('qgw', shape=polymath.wg_dim,     is_sparse=True),
                context_ng_words = C.io.StreamDef('cnw', shape=polymath.wn_dim,     is_sparse=True),
                query_ng_words   = C.io.StreamDef('qnw', shape=polymath.wn_dim,     is_sparse=True),
                answer_begin     = C.io.StreamDef('ab',  shape=polymath.a_dim,      is_sparse=False),
                answer_end       = C.io.StreamDef('ae',  shape=polymath.a_dim,      is_sparse=False),
                context_chars    = C.io.StreamDef('cc',  shape=polymath.word_size,  is_sparse=False),
                query_chars      = C.io.StreamDef('qc',  shape=polymath.word_size,  is_sparse=False),
                is_selected = C.io.StreamDef('sl', shape=1, is_sparse=False))),
        randomize=randomize, randomization_window_in_chunks=1024, randomization_window_in_samples=0,
        max_sweeps=C.io.INFINITELY_REPEAT if repeat else 1)

    input_map = {
        argument_by_name(func, 'cgw'): mb_source.streams.context_g_words,
        argument_by_name(func, 'qgw'): mb_source.streams.query_g_words,
        argument_by_name(func, 'cnw'): mb_source.streams.context_ng_words,
        argument_by_name(func, 'qnw'): mb_source.streams.query_ng_words,
        argument_by_name(func, 'cc' ): mb_source.streams.context_chars,
        argument_by_name(func, 'qc' ): mb_source.streams.query_chars,
        argument_by_name(func, 'ab' ): mb_source.streams.answer_begin,
        argument_by_name(func, 'ae' ): mb_source.streams.answer_end,
        argument_by_name(func, 'sl'): mb_source.streams.is_selected
    }
    return mb_source, input_map
def create_tsv_reader(func, tsv_file, polymath, seqs, num_workers, is_test=False, misc=None):
    with open(tsv_file, 'r', encoding='utf-8') as f:
        eof = False
        batch_count = 0
        while not(eof and (batch_count % num_workers) == 0):
            batch_count += 1
            batch={'cwids':[], 'qwids':[], 'baidx':[], 'eaidx':[], 'ccids':[], 'qcids':[]}

            while not eof and len(batch['cwids']) < seqs: # 读取batch
                line = f.readline()
                if not line:
                    eof = True
                    break

                if misc is not None:
                    import re
                    misc['uid'].append(re.match('^([^\t]*)', line).groups()[0])

                ctokens, qtokens, atokens, cwids, qwids,  baidx, eaidx, ccids, qcids, select = tsv2ctf.tsv_iter(line, polymath.vocab, polymath.chars, is_test, misc)

                batch['cwids'].append(cwids)
                batch['qwids'].append(qwids)
                batch['baidx'].append(baidx)
                batch['eaidx'].append(eaidx)
                batch['ccids'].append(ccids)
                batch['qcids'].append(qcids)
                batch['select'].append(select)

            if len(batch['cwids']) > 0:
                context_g_words  = C.Value.one_hot([[C.Value.ONE_HOT_SKIP if i >= polymath.wg_dim else i for i in cwids] for cwids in batch['cwids']], polymath.wg_dim)
                context_ng_words = C.Value.one_hot([[C.Value.ONE_HOT_SKIP if i < polymath.wg_dim else i - polymath.wg_dim for i in cwids] for cwids in batch['cwids']], polymath.wn_dim)
                query_g_words    = C.Value.one_hot([[C.Value.ONE_HOT_SKIP if i >= polymath.wg_dim else i for i in qwids] for qwids in batch['qwids']], polymath.wg_dim)
                query_ng_words   = C.Value.one_hot([[C.Value.ONE_HOT_SKIP if i < polymath.wg_dim else i - polymath.wg_dim for i in qwids] for qwids in batch['qwids']], polymath.wn_dim)
                context_chars = [np.asarray([[[c for c in cc+[0]*max(0,polymath.word_size-len(cc))]] for cc in ccid], dtype=np.float32) for ccid in batch['ccids']]
                query_chars   = [np.asarray([[[c for c in qc+[0]*max(0,polymath.word_size-len(qc))]] for qc in qcid], dtype=np.float32) for qcid in batch['qcids']]
                answer_begin = [np.asarray(ab, dtype=np.float32) for ab in batch['baidx']]
                answer_end   = [np.asarray(ae, dtype=np.float32) for ae in batch['eaidx']]
                select = [np.asarray(ss,dtype=np.float32) for ss in batch['select']]

                yield { argument_by_name(func, 'cgw'): context_g_words,
                        argument_by_name(func, 'qgw'): query_g_words,
                        argument_by_name(func, 'cnw'): context_ng_words,
                        argument_by_name(func, 'qnw'): query_ng_words,
                        argument_by_name(func, 'cc' ): context_chars,
                        argument_by_name(func, 'qc' ): query_chars,
                        argument_by_name(func, 'ab' ): answer_begin,
                        argument_by_name(func, 'ae' ): answer_end,
                        argument_by_name(func, 'sl'): select }
            else:
                yield {} # need to generate empty batch for distributed training
import pprint
#C.logging.set_trace_level(C.logging.TraceLevel.Info)
def train(data_path, model_path, log_file, config_file, restore=False, profiling=False, gen_heartbeat=False):
    training_config = importlib.import_module(config_file).training_config
    # config for using multi GPUs
    if training_config['multi_gpu']:
        gpu_pad = training_config['gpu_pad']
        gpu_cnt = training_config['gpu_cnt']
        my_rank = C.Communicator.rank()
        my_gpu_id = (my_rank+gpu_pad)%gpu_cnt
        print("rank = "+str(my_rank)+", using gpu "+str(my_gpu_id)+" of "+str(gpu_cnt))
        C.try_set_default_device(C.gpu(my_gpu_id))
    else:
        C.try_set_default_device(C.gpu(3))
    # outputs while training
    normal_log = os.path.join(data_path,training_config['logdir'],log_file)
    # tensorboard files' dir
    tensorboard_logdir = os.path.join(data_path,training_config['logdir'],log_file)

    polymath = PolyMath(config_file)
    z, loss, metric = polymath.model()

    max_epochs = training_config['max_epochs']
    log_freq = training_config['log_freq']

    progress_writers = [C.logging.ProgressPrinter(
                            num_epochs = max_epochs,
                            freq = log_freq,
                            tag = 'Training',
                            log_to_file = normal_log,
                            rank = C.Communicator.rank(),
                            gen_heartbeat = gen_heartbeat)]
    # add tensorboard writer for visualize
    tensorboard_writer = C.logging.TensorBoardProgressWriter(
                             freq=10,
                             log_dir=tensorboard_logdir,
                             rank = C.Communicator.rank(),
                             model = z)
    progress_writers.append(tensorboard_writer)

    lr = C.learning_parameter_schedule(training_config['lr'], minibatch_size=None, epoch_size=None)

    ema = {}
    dummies = []
    for p in z.parameters:
        ema_p = C.constant(0, shape=p.shape, dtype=p.dtype, name='ema_%s' % p.uid)
        ema[p.uid] = ema_p
        dummies.append(C.reduce_sum(C.assign(ema_p, 0.999 * ema_p + 0.001 * p)))
    dummy = C.combine(dummies)

    schedule = C.learning_parameter_schedule([(1,1),(1500,0.1),(5000,0.01),(10000,0.01)])
    learner = C.adadelta(z.parameters, schedule)

    if C.Communicator.num_workers() > 1:
        learner = C.data_parallel_distributed_learner(learner)

    trainer = C.Trainer(z, (loss, metric), learner, progress_writers)

    if profiling:
        C.debugging.start_profiler(sync_gpu=True)

    train_data_file = os.path.join(data_path, training_config['train_data'])
    train_data_ext = os.path.splitext(train_data_file)[-1].lower()

    model_file = os.path.join(model_path, model_name)
    model = C.combine(list(z.outputs) + [loss.output])
    label_ab = argument_by_name(loss, 'ab')

    epoch_stat = {
        'best_val_err' : 100,
        'best_since'   : 0,
        'val_since'    : 0,
        'record_num'   : 0}

    if restore and os.path.isfile(model_file):
        trainer.restore_from_checkpoint(model_file)
        #after restore always re-evaluate
        epoch_stat['best_val_err'] = validate_model(os.path.join(data_path, training_config['val_data']), model, polymath, config_file)

    def post_epoch_work(epoch_stat):
        trainer.summarize_training_progress()
        epoch_stat['val_since'] += 1

        if epoch_stat['val_since'] == training_config['val_interval']:
            epoch_stat['val_since'] = 0
            temp = dict((p.uid, p.value) for p in z.parameters)
            for p in trainer.model.parameters:
                p.value = ema[p.uid].value
            val_err = validate_model(os.path.join(data_path, training_config['val_data']), model, polymath, config_file)
            if epoch_stat['best_val_err'] > val_err:
                epoch_stat['best_val_err'] = val_err
                epoch_stat['best_since'] = 0
                os.system("ls -la >> log.log")
                os.system("ls -la ./Models >> log.log")
                save_flag = True
                fail_cnt = 0
                while save_flag:
                    if fail_cnt > 100:
                        print("ERROR: failed to save models")
                        break
                    try:
                        trainer.save_checkpoint(model_file)
                        epoch_stat['record_num']+=1
                        record_file = os.path.join(model_path,str(epoch_stat['record_num'])+'-'+model_name)
                        trainer.save_checkpoint(record_file)
                        save_flag = False
                    except:
                        fail_cnt = fail_cnt + 1
                for p in trainer.model.parameters:
                    p.value = temp[p.uid]
            else:
                epoch_stat['best_since'] += 1
                if epoch_stat['best_since'] > training_config['stop_after']:
                    return False

        if profiling:
            C.debugging.enable_profiler()

        return True

    if train_data_ext == '.ctf':
        mb_source, input_map = create_mb_and_map(loss, train_data_file, polymath)
        # model = polymath.debug()
        # mb_source, input_map = create_mb_and_map(model,train_data_file, polymath)
        minibatch_size = training_config['minibatch_size'] # number of samples
        epoch_size = training_config['epoch_size']
        # data = mb_source.next_minibatch(minibatch_size,input_map=input_map)
        # res = loss.eval(data)
        # print(res)

        for epoch in range(max_epochs):
            num_seq = 0
            while True:
                if trainer.total_number_of_samples_seen >= training_config['distributed_after']:
                    data = mb_source.next_minibatch(minibatch_size*C.Communicator.num_workers(), input_map=input_map, num_data_partitions=C.Communicator.num_workers(), partition_index=C.Communicator.rank())
                else:
                    data = mb_source.next_minibatch(minibatch_size, input_map=input_map)

                #for k,v in data.items():
                #    print('{}:{}'.format(k, v.data.shape))
                trainer.train_minibatch(data)
                num_seq += trainer.previous_minibatch_sample_count
                # pprint(dummy.eval())
                if num_seq >= epoch_size:
                    break
            if not post_epoch_work(epoch_stat):
                break
    else:
        raise Exception("Unsupported format")

    if profiling:
        C.debugging.stop_profiler()

def symbolic_best_span(begin, end):
    # 获得当前最大值，作为begin+end的预测分数
    max_begin = C.layers.Fold(C.element_max, initial_state=-float("inf"))(begin)
    max_end = C.layers.Fold(C.element_max, initial_state=-float("inf"))(end)
    return max_begin+max_end # [#][1]

def validate_model(test_data, model, polymath,config_file):
    training_config = importlib.import_module(config_file).training_config
    begin_logits = model.outputs[0]
    end_logits   = model.outputs[1]
    cls_score    = model.outputs[2]
    loss         = model.outputs[3]
    root = C.as_composite(loss.owner)
    mb_source, input_map = create_mb_and_map(root, test_data, polymath, randomize=False, repeat=False)
    begin_label = argument_by_name(root, 'ab')
    end_label   = argument_by_name(root, 'ae')

    # 根据预测结果算loss
    begin_prediction = C.sequence.input_variable(1, sequence_axis=begin_label.dynamic_axes[1], needs_gradient=True)
    end_prediction = C.sequence.input_variable(1, sequence_axis=end_label.dynamic_axes[1], needs_gradient=True)
    cls_prediction = C.input_variable(1)

    best_span_score = symbolic_best_span(begin_prediction, end_prediction)
    predicted_span = C.layers.Recurrence(C.plus)(begin_prediction - C.sequence.past_value(end_prediction))
    true_span = C.layers.Recurrence(C.plus)(begin_label - C.sequence.past_value(end_label))
    common_span = C.element_min(predicted_span, true_span)
    begin_match = C.sequence.reduce_sum(C.element_min(begin_prediction, begin_label))
    end_match = C.sequence.reduce_sum(C.element_min(end_prediction, end_label))

    predicted_len = C.sequence.reduce_sum(predicted_span)
    true_len = C.sequence.reduce_sum(true_span)
    common_len = C.sequence.reduce_sum(common_span)
    f1 = 2*common_len/(predicted_len+true_len)
    exact_match = C.element_min(begin_match, end_match)
    precision = common_len/predicted_len
    recall = common_len/true_len
    overlap = C.greater(common_len, 0)
    s = lambda x: C.reduce_sum(x, axis=C.Axis.all_axes())
    stats = C.splice(s(f1), s(exact_match), s(precision), s(recall), s(overlap), s(begin_match), s(end_match))

    # Evaluation parameters
    minibatch_size = 1024 # int(training_config['minibatch_size']/4)
    num_sequences = 0

    stat_sum = 0
    loss_sum = 0


    while True:
        data = mb_source.next_minibatch(minibatch_size, input_map=input_map)
        if not data or not (begin_label in data) or data[begin_label].num_sequences == 0:
            break
        out = model.eval(data, outputs=[begin_logits,end_logits,cls_score,loss], as_numpy=False)
        testloss = out[loss]
        g = best_span_score.grad({begin_prediction:out[begin_logits], end_prediction:out[end_logits]}, wrt=[begin_prediction,end_prediction], as_numpy=False)
        other_input_map = {begin_prediction: g[begin_prediction], end_prediction: g[end_prediction], begin_label: data[begin_label], end_label: data[end_label]}
        stat_sum += stats.eval((other_input_map))
        loss_sum += np.sum(testloss.asarray())
        num_sequences += data[begin_label].num_sequences

    stat_avg = stat_sum / num_sequences
    loss_avg = loss_sum / num_sequences

    print("Validated {} sequences, loss {:.4f}, F1 {:.4f}, EM {:.4f}, precision {:4f}, recall {:4f} hasOverlap {:4f}, start_match {:4f}, end_match {:4f}".format(
            num_sequences,
            loss_avg,
            stat_avg[0],
            stat_avg[1],
            stat_avg[2],
            stat_avg[3],
            stat_avg[4],
            stat_avg[5],
            stat_avg[6]))

    return loss_avg

# map from token to char offset
def w2c_map(s, words):
    w2c=[]
    rem=s
    offset=0
    for i,w in enumerate(words):
        cidx=rem.find(w)
        assert(cidx>=0)
        w2c.append(cidx+offset)
        offset+=cidx + len(w)
        rem=rem[cidx + len(w):]
    return w2c

# get phrase from string based on tokens and their offsets
def get_answer(raw_text, tokens, start, end):
    try:
        w2c=w2c_map(raw_text, tokens)
        return raw_text[w2c[start]:w2c[end]+len(tokens[end])]
    except:
        import pdb
        pdb.set_trace()

def test(test_data, model_path, model_file, config_file):
    training_config = importlib.import_module(config_file).training_config
    # config for using multi GPUs
    if training_config['multi_gpu']:
        gpu_pad = training_config['gpu_pad']
        gpu_cnt = training_config['gpu_cnt']
        my_rank = C.Communicator.rank()
        my_gpu_id = (my_rank+gpu_pad)%gpu_cnt
        print("rank = "+str(my_rank)+", using gpu "+str(my_gpu_id)+" of "+str(gpu_cnt))
        C.try_set_default_device(C.gpu(my_gpu_id))
    else:
        C.try_set_default_device(C.gpu(0))
    polymath = PolyMath(config_file)
    model = C.load_model(os.path.join(model_path, model_file if model_file else model_name))
    begin_logits = model.outputs[0]
    end_logits   = model.outputs[1]
    cls_scores   = model.outputs[2]
    loss         = C.as_composite(model.outputs[3].owner)
    begin_prediction = C.sequence.input_variable(1, sequence_axis=begin_logits.dynamic_axes[1], needs_gradient=True)
    end_prediction = C.sequence.input_variable(1, sequence_axis=end_logits.dynamic_axes[1], needs_gradient=True)
    cls_prediction = C.input_variable(1) # [#][1]
    best_span_score = symbolic_best_span(begin_prediction, end_prediction)
    # 开始-0+0-结束=开始-结束 即长度
    # predicted_span = C.layers.Recurrence(C.plus)(begin_prediction - C.sequence.past_value(end_prediction))

    batch_size = 32 # in sequences
    misc = {'rawctx':[], 'ctoken':[], 'answer':[], 'uid':[]}
    tsv_reader = create_tsv_reader(loss, test_data, polymath, batch_size, 1, is_test=True, misc=misc)
    results = {}
    with open('{}_out.json'.format(model_file), 'w', encoding='utf-8') as json_output:
        for data in tsv_reader:
            out = model.eval(data, outputs=[begin_logits,end_logits,cls_scores,loss], as_numpy=False)
            # 计算正负例, 大于0.5为正
            cls_res = C.greater(cls_prediction,C.constant(0.5)).eval({cls_prediction:out[cls_scores]}) # [#][1]
            # 计算梯度，只有被选为最大的那2个位置梯度为1 # [#,c][1]
            g = best_span_score.grad({begin_prediction:out[begin_logits], end_prediction:out[end_logits]}, wrt=[begin_prediction,end_prediction], as_numpy=False)
            begin_res, end_res = g[begin_prediction], g[end_prediction]
            span = begin_res+end_res
            print(span);return
            # 去掉预测为负例的结果
            for seq, (raw_text, ctokens, answer, uid) in enumerate(zip(misc['rawctx'], misc['ctoken'], misc['answer'], misc['uid'])):
                if cls_res[seq] <= 0:
                    continue
                seq_where = np.argwhere(span[seq])[:,0]
                span_begin = np.min(seq_where)
                span_end = np.max(seq_where)
                predict_answer = get_answer(raw_text, ctokens, span_begin, span_end)
                results['query_id'] = int(uid)
                results['answers'] = [predict_answer]
                json.dump(results, json_output)
                json_output.write("\n")
            misc['rawctx'] = []
            misc['ctoken'] = []
            misc['answer'] = []
            misc['uid'] = []

if __name__=='__main__':
    # default Paths relative to current python file.
    abs_path   = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(abs_path, 'Models')
    data_path  = os.path.join(abs_path, '.')

    parser = argparse.ArgumentParser()
    parser.add_argument('-datadir', '--datadir', help='Data directory where the dataset is located', required=False, default=data_path)
    parser.add_argument('-outputdir', '--outputdir', help='Output directory for checkpoints and models', required=False, default=None)
    parser.add_argument('-logfile', '--logfile', help='Log file version', required=False, default=log_file_name)
    parser.add_argument('-profile', '--profile', help="Turn on profiling", action='store_true', default=False)
    parser.add_argument('-genheartbeat', '--genheartbeat', help="Turn on heart-beat for philly", action='store_true', default=False)
    parser.add_argument('-config', '--config', help='Config file', required=False, default='config')
    parser.add_argument('-r', '--restart', help='Indicating whether to restart from scratch (instead of restart from checkpoint file by default)', action='store_true')
    parser.add_argument('-test', '--test', help='Test data file', required=False, default=None)
    parser.add_argument('-model', '--model', help='Model file name', required=False, default=model_name)

    args = vars(parser.parse_args())

    if args['outputdir'] is not None:
        model_path = args['outputdir'] + "/models"
    if args['datadir'] is not None:
        data_path = args['datadir']
        
    #C.try_set_default_device(C.gpu(0))

    test_data = args['test']
    test_model = args['model']
    if test_data:
        test(test_data, model_path, test_model, args['config'])
    else:
        try:
            train(data_path, model_path, args['logfile'], args['config'],
                restore = not args['restart'],
                profiling = args['profile'],
                gen_heartbeat = args['genheartbeat'])
        finally:
            C.Communicator.finalize()
