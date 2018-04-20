import cntk as C
from cntk.layers import *
from helpers import *
import polymath
import importlib
# =============== factory function ==============
def create_birnn(runit_forward,runit_backward, name=''):
    with C.layers.default_options(initial_state=0.1):
        negRnn = C.layers.Recurrence(runit_backward, go_backwards=True)
        posRnn = C.layers.Recurrence(runit_forward, go_backwards=False)
    @C.Function
    def BiRnn(e):
        h = C.splice(posRnn(e), negRnn(e), name=name)
        return h
    return BiRnn
# ============== class =================
class RNet(polymath.PolyMath):
    def __init__(self, config_file):
        super(RNet,self).__init__(config_file)
        data_config = importlib.import_module(config_file).data_config
        model_config = importlib.import_module(config_file).model_config

        self.convs = model_config['char_convs']
        self.highway_layers = model_config['highway_layers']

    def charcnn(self, x):
        conv_out = C.layers.Sequential([
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

        qce = C.one_hot(qc_ph, num_classes=self.c_dim, sparse_output=self.use_sparse)
        cce = C.one_hot(cc_ph, num_classes=self.c_dim, sparse_output=self.use_sparse)
        word_embed = self.word_glove()(input_glove_words, input_nonglove_words)
        char_embed = self.char_glove()(input_chars)
        embedded = C.splice(word_embed, C.reshape(self.charcnn(char_embed),self.convs), name='splice_embeded')

        highway = HighwayNetwork(dim=self.word_emb_dim+self.convs, highway_layers=self.highway_layers)(embedded)
        highway_drop = C.layers.Dropout(self.dropout)(highway)
        processed = OptimizedRnnStack(self.hidden_dim, bidirectional=True, use_cudnn=self.use_cudnn, name='input_rnn')(highway_drop)

        q_processed = processed.clone(C.CloneMethod.share, {input_chars:qce, input_glove_words:qgw_ph, input_nonglove_words:qnw_ph})
        c_processed = processed.clone(C.CloneMethod.share, {input_chars:cce, input_glove_words:cgw_ph, input_nonglove_words:cnw_ph})

        return C.as_block(
            C.combine([c_processed, q_processed]),
            [(cgw_ph, cgw),(cnw_ph, cnw),(cc_ph, cc),(qgw_ph, qgw),(qnw_ph, qnw),(qc_ph, qc)],
            'input_layer',
            'input_layer')

    def dot_attention(self,inputs, memory):
        '''
        @inputs: [#,c][d] a sequence need attention
        @memory(key): [#,q][d] a sequence input refers to compute similarity(weight)
        @value: [#,q][d] a sequence input refers to weighted sum
        @output: [#,c][d] attention vector
        '''
        input_ph = C.placeholder()
        input_mem = C.placeholder()
        with C.layers.default_options(bias=False, activation=C.relu): # all the projections have no bias
            attn_proj_enc = C.layers.Dense(self.hidden_dim, init=glorot_uniform(), input_rank=1, name="Wqu")
            attn_proj_dec = C.layers.Dense(self.hidden_dim, init=glorot_uniform(), input_rank=1)

        inputs_ = attn_proj_enc(input_ph) # [#,c][d]
        memory_ = attn_proj_dec(input_mem) # [#,q][d]
        unpack_memory, mem_mask = C.sequence.unpack(memory_, 0).outputs # [#][*=q, d], [#][*=q]
        unpack_memory_expand = C.sequence.broadcast_as(unpack_memory, inputs_) # [#,c][*=q,d]

        matrix = C.times_transpose(inputs_, unpack_memory_expand)/(self.hidden_dim**0.5) # [#,c][*=q]
        mem_mask_expand = C.sequence.broadcast_as(mem_mask, inputs_) # [#,c][*=q]
        matrix = C.element_select(mem_mask_expand, matrix, C.constant(-1e+30)) # [#,c][*=q]
        logits = C.reshape(C.softmax(matrix),(-1,1)) # [#,c][*=q,1]
        # [#,c][*=q, d]
        memory_expand = C.sequence.broadcast_as(C.sequence.unpack(input_mem, 0,no_mask_output=True), input_ph)
        weighted_att = C.reshape(C.reduce_sum(logits*memory_expand, axis=0),(-1,)) # [#,c][d]

        return C.as_block(
            C.combine(weighted_att, logits),
            [(input_ph, inputs), (input_mem, memory)],
            'dot attention',
            'dot attention'
        )

    def simi_attention(self, input, memory):
        '''
        return:
        memory weighted vectors over input [#,c][d]
        weight
        '''
        input_ph = C.placeholder() # [#,c][d]
        mem_ph = C.placeholder() # [#,q][d]
        
        input_dense = Dense(2*self.hidden_dim, bias=False,input_rank=1)
        mem_dense = Dense(2*self.hidden_dim, bias=False,input_rank=1)
        bias = C.Parameter(shape=(2*self.hidden_dim,), init=0.0)
        weight_dense = Dense(1,bias=False, input_rank=1)

        proj_inp = input_dense(input_ph) # [#,c][d]
        proj_mem = mem_dense(mem_ph) # [#,q][d]
        unpack_memory, mem_mask = C.sequence.unpack(proj_mem, 0).outputs # [#][*=q, d] [#][*=q]
        expand_mem = C.sequence.broadcast_as(unpack_memory, proj_inp) # [#,c][*=q,d]
        expand_mask = C.sequence.broadcast_as(mem_mask, proj_inp) # [#,c][*=q]
        matrix = C.reshape( weight_dense(C.tanh(proj_inp + expand_mem + bias)) , (-1,)) # [#,c][*=q]
        matrix = C.element_select(expand_mask, matrix, -1e30)
        logits = C.softmax(matrix, axis=0) # [#,c][*=q]
        weight_mem = C.reduce_sum(C.reshape(logits, (-1,1))*expand_mem, axis=0) # [#,c][d]
        weight_mem = C.reshape(weight_mem, (-1,))

        return C.as_block(
            C.combine(weight_mem, logits),
            [(input_ph, input),(mem_ph, memory)],
            'simi_attention','simi_attention'
        )
        
    def gate_attention_layer(self, inputs, memory):
        # [#,c][2*d] [#,c][*=q,1]
        # qc_attn, attn_weight = self.dot_attention(inputs, memory).outputs
        qc_attn, attn_weight = self.simi_attention(inputs, memory).outputs
        cont_attn = C.splice(inputs, qc_attn) # [#,c][4*d]

        dense = Dropout(self.dropout) >> Dense(4*self.hidden_dim, activation=C.sigmoid, input_rank=1) >> Label('gate')
        gate = dense(cont_attn) # [#, c][4*d]
        return gate*cont_attn, attn_weight

    def reasoning_layer(self, inputs, input_dim):
        input_ph = C.placeholder(shape=(input_dim,))
        rnn = create_birnn(GRU(self.hidden_dim), GRU(self.hidden_dim),'reasoning_gru')
        block = Sequential([
                LayerNormalization(name='layerbn'), Dropout(self.dropout), rnn
            ])
        res = block(input_ph)
        return C.as_block(
            res,[(input_ph, inputs)], 'reasoning layer', 'reasoning layer'
        )

    def weighted_sum(self, inputs):
        input_ph = C.placeholder()
        weight = Sequential([
            BatchNormalization(),
            Dropout(self.dropout), Dense(self.hidden_dim, activation=C.tanh),
            Dense(1,bias=False),
            C.sequence.softmax
        ])(input_ph) # [#,c][1]
        res = C.sequence.reduce_sum(weight*input_ph)
        return C.as_block(C.combine(res, weight),
            [(input_ph, inputs)], 'weighted sum','weighted sum')

    def output_layer(self, init, memory):

        def pointer(inputs, state):
            input_ph = C.placeholder()
            state_ph = C.placeholder()
            state_expand = C.sequence.broadcast_as(state_ph, input_ph)
            weight = Sequential([
                BatchNormalization(),
                Dropout(self.dropout),
                Dense(self.hidden_dim, activation=C.sigmoid),Dense(1,bias=False),
                C.sequence.softmax
            ])(C.splice(input_ph, state_expand))
            res = C.sequence.reduce_sum(weight*input_ph)
            return C.as_block(
                C.combine(res, weight),
                [(input_ph, inputs), (state_ph, state)],
                'pointer', 'pointer')

        gru = GRU(2*self.hidden_dim)
        inp, logits1 = pointer(memory, init).outputs
        state2 = gru(init, inp)
        logits2 = pointer(memory, state2).outputs[1]

        return logits1, logits2

    def match_layer(self, query, doc):
        '''
        judge if answer come from this layer
        '''
        qry_ph = C.placeholder()
        doc_ph = C.placeholder()

        dense = Dense(2*self.hidden_dim, C.relu)
        classifier = C.layers.Sequential([
            C.layers.Dropout(self.dropout),
            OptimizedRnnStack(self.hidden_dim, bidirectional=True, use_cudnn=self.use_cudnn, name='model_rnn2'),
            C.sequence.last,
            Dense(1,C.sigmoid)
            ])
        proj_qry = dense(qry_ph)
        c2q_context,simi_weight = self.simi_attention(proj_qry, doc_ph).outputs
        res = C.reshape(classifier(C.splice(proj_qry, c2q_context)),(-1,))
        return C.as_block(
            res,
            [(qry_ph,query),(doc_ph, doc)],
            'match layer','match layer'
        )
        
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
        slc = C.input_variable(1, name='sl')
        input_phs = {'cgw':cgw, 'cnw':cnw, 'qgw':qgw, 'qnw':qnw,
                     'cc':cc, 'qc':qc, 'ab':ab, 'ae':ae, 'sl':slc}
        self._input_phs = input_phs

        # graph
        pu, qu = self.input_layer(cgw, cnw, cc, qgw, qnw, qc).outputs
        gate_pu, wei1 = self.gate_attention_layer(pu, qu) # [#,c][4*hidden]
        self.info['attn1'] = wei1
        print('[RNet build]gate_pu:{}'.format(gate_pu))
        pv = self.reasoning_layer(gate_pu, 4*self.hidden_dim) # [#,c][2*hidden]
        cls_logits = self.match_layer(qu, pv) # [#][1]
        cls_mask = 1.0 - C.greater_equal(cls_logits,[0.5])

        gate_self, wei2 = self.gate_attention_layer(pv,pv) # [#,c][4*hidden]
        self.attn2['attn2'] = wei2
        ph = self.reasoning_layer(gate_self, 4*self.hidden_dim) # [#,c][2*hidden]
        init_pu = self.weighted_sum(pu)

        start_logits, end_logits  = self.output_layer(init_pu.outputs[0], ph) # [#, c][1]
        # scale mask
        expand_cls_logits = C.sequence.broadcast_as(mod_cls_logits,start_logits)
        logits_flag=C.element_select(C.sequence.is_first(start_logits), expand_cls_logits, 1-expand_cls_logits)
        start_logits = start_logits/logits_flag
        end_logits = end_logits/logits_flag 

        # loss
        cls_loss = focal_loss(cls_logits,slc)
        # span loss [#][1] + cls loss [#][1]
        new_loss = all_spans_loss(start_logits, ab, end_logits, ae) + C.constant(10)*cls_loss

        metric = C.classification_error(cls_logits, slc)
        res = C.combine([start_logits, end_logits, cls_logits])
        self._model = res
        self._loss = new_loss
        self._metric = metric
        return self._model, self._loss, self._input_phs

class RNetFeature(RNet):
    def __init__(self, config_file):
        super(RNetFeature, self).__init__(config_file)
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
        slc = C.input_variable(1, name='sl')
        qf = C.input_variable(1, dynamic_axes=[b,q], is_sparse=False, name='query_feature')
        df = C.input_variable(3, dynamic_axes=[b,c], is_sparse=False, name='doc_feature')
        input_phs = {'cgw':cgw, 'cnw':cnw, 'qgw':qgw, 'qnw':qnw,
                     'cc':cc, 'qc':qc, 'ab':ab, 'ae':ae, 'sl':slc,
                     'qf':qf, 'df':df}
        self._input_phs = input_phs
        # graph
        pu, qu = self.input_layer(cgw, cnw, cc, qgw, qnw, qc).outputs
        qu = C.splice(qu, qf); pu = C.splice(pu, df)
        gate_pu, wei1 = self.gate_attention_layer(pu, qu) # [#,c][4*hidden]
        self.info['attn1'] = wei1

        pv = self.reasoning_layer(gate_pu, 4*self.hidden_dim) # [#,c][2*hidden]
        pv = C.splice(pv, df)
        cls_logits = self.match_layer(qu, pv) # [#][1]

        gate_self, wei2 = self.gate_attention_layer(pv,pv) # [#,c][4*hidden]
        self.attn2['attn2'] = wei2
        ph = self.reasoning_layer(gate_self, 4*self.hidden_dim) # [#,c][2*hidden]
        init_pu = self.weighted_sum(pu)

        start_logits, end_logits  = self.output_layer(init_pu.outputs[0], ph) # [#, c][1]
        # scale mask
        expand_cls_logits = C.sequence.broadcast_as(cls_logits,start_logits)
        logits_flag=C.element_select(C.sequence.is_first(start_logits), expand_cls_logits, 1-expand_cls_logits)
        start_logits = start_logits/logits_flag
        end_logits = end_logits/logits_flag 

        # loss
        cls_loss = focal_loss(cls_logits,slc)
        # span loss [#][1] + cls loss [#][1]
        new_loss = all_spans_loss(start_logits, ab, end_logits, ae) + cls_loss

        metric = C.classification_error(cls_logits, slc)
        res = C.combine([start_logits, end_logits, cls_logits])
        self._model = res
        self._loss = new_loss
        self._metric = metric
        return self._model, self._loss, self._input_phs

# =============== test edition ==================
from cntk.debugging import debug_model
def test_model_part():
    from train_pm import  create_mb_and_map
    rnet = RNet('config')
    model,loss, input_phs = rnet.build_model()
    mb, input_map = create_mb_and_map(input_phs, 'dev.ctf', rnet)
    data=mb.next_minibatch(3,input_map=input_map)
    res = model.eval(data)
    print(res)
def _testcode():
    data=[np.array([[1,2,3,0],[1,2,3,0]]),
        np.array([[1,2,0,0],[2,3,0,0]]),
        np.array([[4,0,0,0],[5,0,0,0],[6,0,0,0]])]
    inp=C.sequence.input_variable(4)

if __name__ == '__main__':
    C.try_set_default_device(C.gpu(2))
    test_model_part()
