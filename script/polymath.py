import cntk as C
import numpy as np
from helpers import *
import pickle
import importlib
import os

class PolyMath(object):
    def __init__(self, config_file):
        data_config = importlib.import_module(config_file).data_config
        model_config = importlib.import_module(config_file).model_config

        self.word_count_threshold = data_config['word_count_threshold']
        self.char_count_threshold = data_config['char_count_threshold']
        self.word_size = data_config['word_size']
        self.abs_path = os.path.dirname(os.path.abspath(__file__))
        pickle_file = os.path.join(self.abs_path, data_config['pickle_file'])
        self.word_embed_file = data_config['word_embed_file']
        self.char_embed_file = data_config['char_embed_file']

        with open(pickle_file, 'rb') as vf:
            known, self.vocab, self.chars = pickle.load(vf)

        self.wg_dim = known
        self.wn_dim = len(self.vocab) - known
        self.c_dim = len(self.chars)
        self.a_dim = 1

        self.hidden_dim = model_config['hidden_dim']
        self.dropout = model_config['dropout']
        self.char_emb_dim = model_config['char_emb_dim']
        self.word_emb_dim = model_config['word_emb_dim']
        self.use_cudnn = model_config['use_cudnn']
        self.use_sparse = True

        print('dropout', self.dropout)
        print('use_cudnn', self.use_cudnn)
        print('use_sparse', self.use_sparse)

        self._model=None
        self._loss=None
        self._input_phs=None

    def word_glove(self):
        # load glove
        npglove = np.zeros((self.wg_dim, self.word_emb_dim, dtype=np.float32)
        with open(os.path.join(self.abs_path, self.word_embed_file), encoding='utf-8') as f:
            for line in f:
                parts = line.split()
                word = parts[0].lower()
                if self.vocab.get(word, self.wg_dim)<self.wg_dim:
                    npglove[self.vocab[word],:] = np.asarray([float(p) for p in parts[1:]])
        glove = C.constant(npglove)
        nonglove = C.parameter(shape=(len(self.vocab) - self.wg_dim, self.word_emb_size), init=C.glorot_uniform(), name='TrainableE')
        @C.Function
        def func(wg, wn):
            return C.times(wg, glove) + C.times(wn, nonglove)
        return func
    def char_glove(self):
        npglove = np.zeros(self.c_dim, self.char_emb_dim, dtype=np.float32)
        # only 94 known chars, 308 chars in all
        with open(os.path.join(self.abs_path, self.char_embed_file), encoding='utf-8') as f:
            for line in f:
                parts = line.split()
                word = parts[0].lower()
                if self.chars.get(word, self.c_dim)<self.c_dim:
                    npglove[self.chars[word],:] = np.asarray([float(p) for p in parts[1:]])
        glove = C.constant(npglove)
        @C.Function
        def func(cg):
             return C.times(cg, glove)
         return func
    def build_model(self):
        raise NotImplementedError
    @property
    def model(self):
        if not self._model:
            self._model, self._loss, self._input_phs = self.build_model()
        return self._model
    @property
    def loss(self):
        if not self._model:
            self._model, self._loss, self._input_phs = self.build_model()
        return self._loss
    @property
    def input_phs(self):
        if not self._model:
            self._model, self._loss, self._input_phs = self.build_model()
        return self._input_phs

class BiDAF(PolyMath):
    def __init__(self, config_file):
        super(self, BiDAF).__init__(config_file)
        data_config = importlib.import_module(config_file).data_config
        model_config = importlib.import_module(config_file).model_config

        self.convs = model_config['char_convs']
        self.highway_layers = model_config['highway_layers']
        self.two_step = model_config['two_step']

    def charcnn(self, x):
        conv_out = C.layers.Sequential([
            C.layers.Embedding(self.char_emb_dim),
            C.layers.Dropout(self.dropout),
            C.layers.Convolution2D((5,self.char_emb_dim), self.convs, activation=C.relu, init=C.glorot_uniform(), bias=True, init_bias=0, name='charcnn_conv')])(x)
        return C.reduce_max(conv_out, axis=1) # workaround cudnn failure in GlobalMaxPooling

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
            self.word_glove()(input_glove_words, input_nonglove_words), name='splice_embed')
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

        # This part deserves some explanation
        # It is the attention layer
        # In the paper they use a 6 * dim dimensional vector
        # here we split it in three parts because the different parts
        # participate in very different operations
        # so W * [h; u; h.* u] becomes w1 * h + w2 * u + w3 * (h.*u)
        ws1 = C.parameter(shape=(2 * self.hidden_dim, 1), init=C.glorot_uniform())
        ws2 = C.parameter(shape=(2 * self.hidden_dim, 1), init=C.glorot_uniform())
        ws3 = C.parameter(shape=(1, 2 * self.hidden_dim), init=C.glorot_uniform())
        att_bias = C.parameter(shape=(), init=0)

        wh = C.times (c_processed, ws1)
        wu = C.reshape(C.times (qvw, ws2), (-1,))
        whu = C.reshape(C.reduce_sum(c_processed * C.sequence.broadcast_as(qvw * ws3, c_processed), axis=1), (-1,))
        S = wh + whu + C.sequence.broadcast_as(wu, c_processed) + att_bias
        # mask out values outside of Query, and fill in gaps with -1e+30 as neutral value for both reduce_log_sum_exp and reduce_max
        qvw_mask_expanded = C.sequence.broadcast_as(qvw_mask, c_processed)
        S = C.element_select(qvw_mask_expanded, S, C.constant(-1e+30))
        q_attn = C.reshape(C.softmax(S), (-1,1))
        #q_attn = print_node(q_attn)
        c2q = C.reshape(C.reduce_sum(C.sequence.broadcast_as(qvw, q_attn) * q_attn, axis=0),(-1))

        max_col = C.reduce_max(S)
        c_attn = C.sequence.softmax(max_col)

        htilde = C.sequence.reduce_sum(c_processed * c_attn)
        q2c = C.sequence.broadcast_as(htilde, c_processed)
        q2c_out = c_processed * q2c

        att_context = C.splice(c_processed, c2q, c_processed * c2q, q2c_out)

        return C.as_block(
            att_context,
            [(c_processed, context), (q_processed, query)],
            'attention_layer',
            'attention_layer')

    def modeling_layer(self, attention_context):
        att_context = C.placeholder(shape=(8*self.hidden_dim,))
        #modeling layer
        # todo: use dropout in optimized_rnn_stack from cudnn once API exposes it
        mod_context = C.layers.Sequential([
            C.layers.Dropout(self.dropout),
            OptimizedRnnStack(self.hidden_dim, bidirectional=True, use_cudnn=self.use_cudnn, name='model_rnn0'),
            C.layers.Dropout(self.dropout),
            OptimizedRnnStack(self.hidden_dim, bidirectional=True, use_cudnn=self.use_cudnn, name='model_rnn1')])(att_context)

        return C.as_block(
            mod_context,
            [(att_context, attention_context)],
            'modeling_layer',
            'modeling_layer')

    def output_layer(self, attention_context, modeling_context):
        att_context = C.placeholder(shape=(8*self.hidden_dim,))
        mod_context = C.placeholder(shape=(2*self.hidden_dim,))
        #output layer [#,c][1]
        start_logits = C.layers.Dense(1, name='out_start')(C.dropout(C.splice(mod_context, att_context), self.dropout))

        start_hardmax = seq_hardmax(start_logits) # [000010000]
        att_mod_ctx = C.sequence.last(C.sequence.gather(mod_context, start_hardmax)) # [#][2*hidden_dim]
        att_mod_ctx_expanded = C.sequence.broadcast_as(att_mod_ctx, att_context)
        end_input = C.splice(att_context, mod_context, att_mod_ctx_expanded, mod_context * att_mod_ctx_expanded) # [#, c][14*hidden_dim]
        m2 = OptimizedRnnStack(self.hidden_dim, bidirectional=True, use_cudnn=self.use_cudnn, name='output_rnn')(end_input)
        end_logits = C.layers.Dense(1, name='out_end')(C.dropout(C.splice(m2, att_context), self.dropout))

        return C.as_block(
            C.combine([start_logits, end_logits]),
            [(att_context, attention_context), (mod_context, modeling_context)],
            'output_layer',
            'output_layer')

    def build_model(self):
        c = C.Axis.new_unique_dynamic_axis('c')
        q = C.Axis.new_unique_dynamic_axis('q')
        b = C.Axis.default_batch_axis()
        cgw = C.input_variable(self.wg_dim, dynamic_axes=[b,c], is_sparse=self.use_sparse, name='cgw')
        cnw = C.input_variable(self.wn_dim, dynamic_axes=[b,c], is_sparse=self.use_sparse, name='cnw')
        qgw = C.input_variable(self.wg_dim, dynamic_axes=[b,q], is_sparse=self.use_sparse, name='qgw')
        qnw = C.input_variable(self.wn_dim, dynamic_axes=[b,q], is_sparse=self.use_sparse, name='qnw')
        cc = C.input_variable((1,self.word_size), dynamic_axes=[b,c], name='cc')
        qc = C.input_variable((1,self.word_size), dynamic_axes=[b,q], name='qc')
        ab = C.input_variable(self.a_dim, dynamic_axes=[b,c], name='ab')
        ae = C.input_variable(self.a_dim, dynamic_axes=[b,c], name='ae')
        input_phs = {'cgw':cgw, 'cnw':cnw, 'qgw':qgw, 'qnw':qnw,
                        'cc':cc, 'qc':qc, 'ab':ab, 'ae':ae}
        self._input_phs = input_phs

        #input layer
        c_processed, q_processed = self.input_layer(cgw,cnw,cc,qgw,qnw,qc).outputs

        # attention layer
        att_context = self.attention_layer(c_processed, q_processed)

        # modeling layer
        mod_context = self.modeling_layer(att_context)

        # output layer
        start_logits, end_logits = self.output_layer(att_context, mod_context).outputs

        # loss
        start_loss = seq_loss(start_logits, ab)
        end_loss = seq_loss(end_logits, ae)
        #paper_loss = start_loss + end_loss
        new_loss = all_spans_loss(start_logits, ab, end_logits, ae)
        self._model = C.combine([start_logits,end_logits])
        self._loss = new_loss
        return self._model, self._loss, self._input_phs


from subnetworks import IndRNN
class BiDAFInd(BiDAF):
    def __init__(self, config_file):
        super(self, BiDAFInd).__init__(config_file)
        model_config = importlib.import_module(config_file).model_config

        self._time_step = 100 #TODO: dataset explore
        self._indrnn_builder = IndRNN(self.hidden_dim, self.hidden_dim, recurrent_max_abs=pow(2, 1/self._time_step), activation = C.leaky_relu)
        self.use_layerbn = model_config['use_layerbn']

    def charcnn(self, x):
        '''
        @x:[I,w1,w2,w3,...]
        @kernal:[O,I,w1,w2,w3,...]
        @out:[O,?depend on stride]
        '''
        conv_out = C.layers.Sequential([
            C.layers.Dropout(self.dropout),
            C.layers.Convolution2D((5,self.char_emb_dim), self.convs, activation=C.relu, init=C.glorot_uniform(), bias=True, init_bias=0, name='charcnn_conv')])(x)
        return C.reduce_max(conv_out, axis=1) # workaround cudnn failure in GlobalMaxPooling

    def input_layer(self, x):
        cgw_ph = C.placeholder()
        cnw_ph = C.placeholder()
        cc_ph  = C.placeholder()
        qgw_ph = C.placeholder()
        qnw_ph = C.placeholder()
        qc_ph  = C.placeholder()
        input_chars = C.placeholder(shape=(1,self.word_size,self.c_dim))
        input_glove_words = C.placeholder(shape=(self.wg_dim,))
        input_nonglove_words = C.placeholder(shape=(self.wn_dim,))

        qce = C.one_hot(qc_ph, num_classes=self.c_dim, sparse_output=self.use_sparse)
        cce = C.one_hot(cc_ph, num_classes=self.c_dim, sparse_output=self.use_sparse)
        word_embed = self.word_glove()(input_glove_words, input_nonglove_words)
        char_embed = self.char_glove()(input_chars)
        embeded = C.splice(word_embed, C.reshape(self.charcnn(char_embed),self.convs), name='splice_embeded')

        self._indrnn_builder._input_size = self.word_emb_dim+self.convs
        ind1 = self._indrnn_builder.build()
        self._indrnn_builder._input_size = 2*self.hidden_dim
        indrnns = [self._indrnn_builder.build() for _ in range(5)]
        indrnns.insert(0,ind1)

        process = C.layers.For(
            range(3),lambda i:C.Sequential([
                C.layers.Dropout(self.dropout),
                (C.layers.Recurrence(indrnns[2*i]),C.layers.Recurrence(indrnns[2*i+1],go_backwards=True)),
                C.splice
            ]))
        processed = process(embeded)

        q_processed = processed.clone(C.CloneMethod.share, {input_chars:qce, input_glove_words:qgw_ph, input_nonglove_words:qnw_ph})
        c_processed = processed.clone(C.CloneMethod.share, {input_chars:cce, input_glove_words:cgw_ph, input_nonglove_words:cnw_ph})

        return C.as_block(
            C.combine([c_processed, q_processed]),
            [(cgw_ph, cgw),(cnw_ph, cnw),(cc_ph, cc),(qgw_ph, qgw),(qnw_ph, qnw),(qc_ph, qc)],
            'input_layer',
            'input_layer')


    def modeling_layer(self, attention_context):
        att_context = C.placeholder(shape=(8*self.hidden_dim,))
        self._indrnn_builder._input_size = 8*self.hidden_dim
        ind1 = self._indrnn_builder.build()
        self._indrnn_builder._input_size = 2*self.hidden_dim
        indrnns = [self._indrnn_builder.build() for _ in range(11)]
        indrnns.insert(0,ind1)
        #modeling layer 6 resnet layers
        model = C.layers.For(range(3),lambda i:C.layers.Sequential([
            C.layers.ResNetBlock(
                C.layers.Sequentail([
                    C.layers.LayerNormalization() if self.use_layerbn else C.layers.identity,
                    C.layers.Dropout(self.dropout),
                    (C.layers.Recurrence(indrnns[4*i]), C.layers.Recurrence(indrnns[4*i+1],go_backwards=True)),
                    C.splice,
                    C.layers.LayerNormalization() if self.use_layerbn else C.layers.identity,
                    C.layers.Dropout(self.dropout),
                    (C.layers.Recurrence(indrnns[4*i+2]), C.layers.Recurrence(indrnns[4*i+3],go_backwards=True)),
                    C.splice
                ])
            )
        ]))
        mod_context = model(att_context)
        return C.as_block(
            mod_context,
            [(att_context, attention_context)],
            'modeling_layer',
            'modeling_layer')


