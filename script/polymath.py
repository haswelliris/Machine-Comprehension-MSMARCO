import cntk as C
import numpy as np
from helpers import *
import pickle
import importlib
import os

class PolyMath:
    def __init__(self, config_file):
        data_config = importlib.import_module(config_file).data_config
        model_config = importlib.import_module(config_file).model_config

        self.word_count_threshold = data_config['word_count_threshold']
        self.char_count_threshold = data_config['char_count_threshold']
        self.word_size = data_config['word_size']
        self.abs_path = os.path.dirname(os.path.abspath(__file__))
        pickle_file = os.path.join(self.abs_path, data_config['pickle_file'])

        with open(pickle_file, 'rb') as vf:
            known, self.vocab, self.chars = pickle.load(vf)

        self.wg_dim = known
        self.wn_dim = len(self.vocab) - known
        self.c_dim = len(self.chars)
        self.a_dim = 1

        self.hidden_dim = model_config['hidden_dim']
        self.convs = model_config['char_convs']
        self.dropout = model_config['dropout']
        self.char_emb_dim = model_config['char_emb_dim']
        self.highway_layers = model_config['highway_layers']
        self.two_step = model_config['two_step']
        self.use_cudnn = model_config['use_cudnn']
        self.use_sparse = True
        # self.ldb = model_config['lambda'] # 控制两个loss的平衡

        print('dropout', self.dropout)
        print('use_cudnn', self.use_cudnn)
        print('use_sparse', self.use_sparse)

    def charcnn(self, x):
        conv_out = C.layers.Sequential([
            C.layers.Embedding(self.char_emb_dim),
            C.layers.Dropout(self.dropout),
            C.layers.Convolution2D((5,self.char_emb_dim), self.convs, activation=C.relu, init=C.glorot_uniform(), bias=True, init_bias=0, name='charcnn_conv')])(x)
        return C.reduce_max(conv_out, axis=1) # workaround cudnn failure in GlobalMaxPooling

    def embed(self):
        # load glove
        npglove = np.zeros((self.wg_dim, self.hidden_dim), dtype=np.float32)
        with open(os.path.join(self.abs_path, 'glove.6B.100d.txt'), encoding='utf-8') as f:
            for line in f:
                parts = line.split()
                word = parts[0].lower()
                if word in self.vocab:
                    npglove[self.vocab[word],:] = np.asarray([float(p) for p in parts[1:]])
        glove = C.constant(npglove)
        nonglove = C.parameter(shape=(len(self.vocab) - self.wg_dim, self.hidden_dim), init=C.glorot_uniform(), name='TrainableE')

        def func(wg, wn):
            return C.times(wg, glove) + C.times(wn, nonglove)
        return func

    def input_layer(self,cgw,cnw,cc,qgw,qnw,qc):
        cgw_ph = C.placeholder()
        cnw_ph = C.placeholder()
        cc_ph  = C.placeholder()
        qgw_ph = C.placeholder()
        qnw_ph = C.placeholder()
        qc_ph  = C.placeholder()

        input_chars = C.placeholder(shape=(1,self.word_size,self.c_dim))
        input_glove_words = C.placeholder(shape=(self.wg_dim,))
        input_nonglove_words = C.placeholder(shape=(self.wn_dim,))

        # we need to reshape because GlobalMaxPooling/reduce_max is retaining a trailing singleton dimension
        # todo GlobalPooling/reduce_max should have a keepdims default to False
        embedded = C.splice(
            C.reshape(self.charcnn(input_chars), self.convs),
            self.embed()(input_glove_words, input_nonglove_words), name='splice_embed')
        highway = HighwayNetwork(dim=2*self.hidden_dim, highway_layers=self.highway_layers)(embedded)
        highway_drop = C.layers.Dropout(self.dropout)(highway)
        processed = OptimizedRnnStack(self.hidden_dim, bidirectional=True, use_cudnn=self.use_cudnn, name='input_rnn')(highway_drop)

        qce = C.one_hot(qc_ph, num_classes=self.c_dim, sparse_output=self.use_sparse)
        cce = C.one_hot(cc_ph, num_classes=self.c_dim, sparse_output=self.use_sparse)

        q_processed = processed.clone(C.CloneMethod.share, {input_chars:qce, input_glove_words:qgw_ph, input_nonglove_words:qnw_ph})
        c_processed = processed.clone(C.CloneMethod.share, {input_chars:cce, input_glove_words:cgw_ph, input_nonglove_words:cnw_ph})

        return C.as_block(
            C.combine([c_processed, q_processed]),
            [(cgw_ph, cgw),(cnw_ph, cnw),(cc_ph, cc),(qgw_ph, qgw),(qnw_ph, qnw),(qc_ph, qc)],
            'input_layer',
            'input_layer')

    def attention_layer(self, context, query):
        q_processed = C.placeholder(shape=(2*self.hidden_dim,))
        c_processed = C.placeholder(shape=(2*self.hidden_dim,))

        #convert query's sequence axis to static
        qvw, qvw_mask = C.sequence.unpack(q_processed, padding_value=0).outputs

        # so W * [h; u; h.* u] becomes w1 * h + w2 * u + w4 * (h.*u)
        ws1 = C.parameter(shape=(2 * self.hidden_dim, 1), init=C.glorot_uniform())
        ws2 = C.parameter(shape=(2 * self.hidden_dim, 1), init=C.glorot_uniform())
        ws3 = C.parameter(shape=(2 * self.hidden_dim, 1), init=C.glorot_uniform())
        ws4 = C.parameter(shape=(1, 2 * self.hidden_dim), init=C.glorot_uniform())
        att_bias = C.parameter(shape=(), init=0)

        wh = C.times (c_processed, ws1) # [#,c][1]
        wu = C.reshape(C.times (qvw, ws2), (-1,)) # [#][*]
        # qvw*ws4: [#][*,200], whu:[#,c][*]
        whu = C.reshape(C.reduce_sum(c_processed * C.sequence.broadcast_as(qvw * ws4, c_processed), axis=1), (-1,))
        S1 = wh + C.sequence.broadcast_as(wu, c_processed) + att_bias # [#,c][*]
        qvw_mask_expanded = C.sequence.broadcast_as(qvw_mask, c_processed)
        S1 = C.element_select(qvw_mask_expanded, S1, C.constant(-1e+30))
        q_attn = C.reshape(C.softmax(S1), (-1,1)) # [#,c][*,1]
        c2q = C.reshape(C.reduce_sum(C.sequence.broadcast_as(qvw, q_attn) * q_attn, axis=0),(-1)) # [#,c][200]

        max_col = C.reduce_max(S1) # [#,c][1] 最大的q中的单词
        c_attn = C.sequence.softmax(max_col) # [#,c][1] 对c中的每一个单词做softmax

        htilde = C.sequence.reduce_sum(c_processed * c_attn) # [#][200]
        q2c = C.sequence.broadcast_as(htilde, c_processed) # [#,c][200]
        q2c_out = c_processed * q2c

        hvw, hvw_mask = C.sequence.unpack(c_processed, padding_value=0).outputs
        whh = C.reshape(C.times(c_processed, ws3),(-1,)) # [#][*,1]
        S2 = wh + C.sequence.broadcast_as(whh, c_processed) + att_bias
        hvw_mask_expanded = C.sequence.broadcast_as(hvw_mask, c_processed)
        S2 = C.element_select(hvw_mask_expanded, S2, C.constant(-1e+30))
        hh_attn = C.reshape(C.softmax(S2), (-1,1))
        c2c = C.reshape(C.reduce_sum(C.sequence.broadcast_as(hvw, hh_attn)*hh_attn, axis=0), (-1,))

        # 原始文档，题目表示，文章重点表示，匹配度表示，文章上下文表示
        att_context = C.splice(c_processed, c2q, q2c_out)
        res = C.combine([att_context, c_processed * c2q, c2c])

        return C.as_block( res,
            [(c_processed, context), (q_processed, query)],
            'attention_layer',
            'attention_layer')

    def modeling_layer(self, attention_context_cls, attention_context_reg):
        '''
        在第一遍阅读后，对文章的整体表示
        '''
        ph1 = C.placeholder(shape=(8*self.hidden_dim,))
        ph2 = C.placeholder(shape=(8*self.hidden_dim,))

        att_context = C.placeholder(shape=(8*self.hidden_dim,))
        mod_context = C.layers.Sequential([
            C.layers.Dropout(self.dropout),
            OptimizedRnnStack(self.hidden_dim, bidirectional=True, use_cudnn=self.use_cudnn, name='model_rnn0'),
            C.layers.Dropout(self.dropout),
            OptimizedRnnStack(self.hidden_dim, bidirectional=True, use_cudnn=self.use_cudnn, name='model_rnn1')])(att_context)

        mod_out_cls = mod_context.clone(C.CloneMethod.clone, {att_context: ph1})
        mod_out_reg = mod_context.clone(C.CloneMethod.clone, {att_context: ph2})

        return C.as_block(
            C.combine([mod_out_cls, mod_out_reg]),
            [(ph1, attention_context_cls),(ph2, attention_context_reg)],
            'modeling_layer',
            'modeling_layer')

    def output_layer(self, attention_context, modeling_context):
        att_context = C.placeholder(shape=(8*self.hidden_dim,))
        mod_context = C.placeholder(shape=(2*self.hidden_dim,))
        #output layer
        # 映射 [#,c][1]
        start_logits = C.layers.Dense(1, name='out_start')(C.dropout(C.splice(mod_context, att_context), self.dropout))
        if self.two_step:
            start_hardmax = seq_hardmax(start_logits)
            # 得到最大单词的语义表示 [#][dim]
            att_mod_ctx = C.sequence.last(C.sequence.gather(mod_context, start_hardmax))
        else:
            start_prob = C.softmax(start_logits)
            att_mod_ctx = C.sequence.reduce_sum(mod_context * start_prob)
        att_mod_ctx_expanded = C.sequence.broadcast_as(att_mod_ctx, att_context)
        end_input = C.splice(att_context, mod_context, att_mod_ctx_expanded, mod_context * att_mod_ctx_expanded)
        m2 = OptimizedRnnStack(self.hidden_dim, bidirectional=True, use_cudnn=self.use_cudnn, name='output_rnn')(end_input)
        end_logits = C.layers.Dense(1, name='out_end')(C.dropout(C.splice(m2, att_context), self.dropout))

        return C.as_block(
            C.combine([start_logits, end_logits]),
            [(att_context, attention_context), (mod_context, modeling_context)],
            'output_layer',
            'output_layer')

    def model(self):
        c = C.Axis.new_unique_dynamic_axis('c')
        q = C.Axis.new_unique_dynamic_axis('q')
        b = C.Axis.default_batch_axis()
        cgw = C.input_variable(self.wg_dim, dynamic_axes=[b,c], is_sparse=self.use_sparse, name='cgw')
        cnw = C.input_variable(self.wn_dim, dynamic_axes=[b,c], is_sparse=self.use_sparse, name='cnw')
        qgw = C.input_variable(self.wg_dim, dynamic_axes=[b,q], is_sparse=self.use_sparse, name='qgw')
        qnw = C.input_variable(self.wn_dim, dynamic_axes=[b,q], is_sparse=self.use_sparse, name='qnw')
        cc = C.input_variable(self.word_size, dynamic_axes=[b,c], name='cc')
        qc = C.input_variable(self.word_size, dynamic_axes=[b,q], name='qc')
        ab = C.input_variable(self.a_dim, dynamic_axes=[b,c], name='ab')
        ae = C.input_variable(self.a_dim, dynamic_axes=[b,c], name='ae')
        slc = C.sequence.input_variable(1, name='sl')

        #input layer
        cc = C.reshape(cc, (1,-1)); qc = C.reshape(qc, (1,-1))
        c_processed, q_processed = self.input_layer(cgw,cnw,cc,qgw,qnw,qc).outputs
        # q_int = C.sequence.last(q_processed) # [#][2*hidden_dim]
        # attention layer output:[#,c][8*hidden_dim]
        att_1,att_2,att_3 = self.attention_layer(c_processed, q_processed).outputs
        att_context_cls = C.splice(att_1, att_2)
        att_context_reg = C.splice(att_1, att_3)

        # modeling layer output:[#][2*hidden_dim] [#,c][2*hidden_dim]
        mod_context_cls,  mod_context_reg= self.modeling_layer(att_context_cls, att_context_reg).outputs
        mod_context_cls = C.sequence.last(mod_context_cls)

        # classify
        cls_p = C.layers.Dense(1, activation=C.sigmoid)(mod_context_cls) # [#][1]
        cls_res = C.greater(cls_p, C.constant(0.5))
        # output layer
        start_logits, end_logits = self.output_layer(att_context_reg, mod_context_reg).outputs

        # loss

        # 负数
        slc = C.reshape(C.sequence.last(slc),(-1,)) # [#][1]
        cons_1 = C.constant(1)
        cls_loss = C.binary_cross_entropy(cls_p ,slc, name='classify')
        # span loss [#][1] + cls loss [#][1]
        new_loss = all_spans_loss(start_logits, ab, end_logits, ae)*slc + cls_loss
        new_loss.as_numpy = False

        metric = C.classification_error(cls_res, slc)
        res = C.combine([start_logits, end_logits, cls_p])
        res.as_numpy=False
        return res, new_loss, metric
    def debug(self):
        c = C.Axis.new_unique_dynamic_axis('c')
        q = C.Axis.new_unique_dynamic_axis('q')
        b = C.Axis.default_batch_axis()
        cgw = C.input_variable(self.wg_dim, dynamic_axes=[b,c], is_sparse=self.use_sparse, name='cgw')
        cnw = C.input_variable(self.wn_dim, dynamic_axes=[b,c], is_sparse=self.use_sparse, name='cnw')
        qgw = C.input_variable(self.wg_dim, dynamic_axes=[b,q], is_sparse=self.use_sparse, name='qgw')
        qnw = C.input_variable(self.wn_dim, dynamic_axes=[b,q], is_sparse=self.use_sparse, name='qnw')
        cc = C.input_variable(self.word_size, dynamic_axes=[b,c], name='cc')
        qc = C.input_variable(self.word_size, dynamic_axes=[b,q], name='qc')
        ab = C.input_variable(self.a_dim, dynamic_axes=[b,c], name='ab')
        ae = C.input_variable(self.a_dim, dynamic_axes=[b,c], name='ae')
        slc = C.sequence.input_variable(1, name='sl')

        #input layer
        cc = C.reshape(cc, (1,-1)); qc = C.reshape(qc, (1,-1))
        c_processed, q_processed = self.input_layer(cgw,cnw,cc,qgw,qnw,qc).outputs
        att_1,att_2,att_3 = self.attention_layer(c_processed, q_processed).outputs
        att_context_cls = C.splice(att_1, att_2)
        att_context_reg = C.splice(att_1, att_3)
        # modeling layer output:[#][2*hidden_dim] [#,c][2*hidden_dim]
        mod_context_cls,  mod_context_reg= self.modeling_layer(att_context_cls, att_context_reg).outputs
        mod_context_cls = C.sequence.last(mod_context_cls)


        start_logits, end_logits = self.output_layer(att_context_reg, mod_context_reg).outputs
        # classify
        cls_p = C.layers.Dense(1, activation=C.sigmoid)(mod_context_cls) # [#][1]
        cls_res = C.greater(cls_p, C.constant(0.5))
        # 负数
        slc = C.reshape(C.sequence.last(slc),(-1,)) # [#][1]
        cons_1 = C.constant(1)
        cls_loss = C.binary_cross_entropy(cls_p ,slc, name='classify')
        # span loss [#][1] + cls loss [#][1]
        # 在这里出错
        new_loss = all_spans_loss(start_logits, ab, end_logits, ae)*slc + cls_loss
        return C.combine(end_logits, cls_loss, C.sequence.last(ab), new_loss)

