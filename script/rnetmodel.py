import cntk as C
from cntk.layers import *
from helpers import *
import polymath
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
        super(self, RNet).__init__(config_file)
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

    def dot_attention(self,inputs, memory):
        '''
        @inputs: [#,c][d] a sequence need attention
        @memory(key): [#,q][d] a sequence input refers to compute similarity(weight)
        @value: [#,q][d] a sequence input refers to weighted sum
        @output: [#,c][d] attention vector
        '''
        input_ph = C.placeholder(shape=(2*self.hidden_dim,))
        input_mem = C.placeholder(shape=(2*self.hidden_dim,))
        with C.layers.default_options(bias=False, activation=C.relu): # all the projections have no bias
            attn_proj_enc = C.layers.Dense(self.hidden_dim, init=glorot_uniform(), input_rank=1, name="Wqu")
            attn_proj_dec = C.layers.Dense(self.hidden_dim, init=glorot_uniform(), input_rank=1)

        inputs_ = attn_proj_enc(input_ph) # [#,c][d]
        memory_ = attn_proj_dec(input_mem) # [#,q][d]
        unpack_memory, mem_mask = C.sequence.unpack(memory_, 0).outputs # [#][*=q, d], [#][*=q]
        matrix = C.times_transpose(inputs_, unpack_memory)/(self.hidden_dim**0.5) # [#,c][*=q]
        mem_mask_expand = C.sequence.broadcast_as(mem_mask, inputs_) # [#,c][*=q]
        matrix = C.element_select(mem_mask_expand, matrix, C.constant(-1e+30)) # [#,c][*=q]
        logits = C.softmax(matrix) # [#,c][*=q]
        # [#,c][*=q, d]
        memory_expand = C.seqence.broadcast_as(C.sequence.unpack(input_mem, 0,no_mask_output=True), inputs)
        weighted_att = C.reshape(C.times(logits, memory_expand),(-1,)) # [#,c][d]

        return C.as_block(
            weighted_att,
            [(input_ph, inputs), (input_mem, memory)],
            'dot attention',
            'dot attention'
        )

    def gate_attention_layer(self, inputs, memory):
        qc_attn = dot_attention(inputs, memory) # [#,c][2*d]
        cont_attn = C.splice(inputs, qc_attn) # [#,c][4*d]

        dense = Dropout(self.dropout) >> Dense(4*self.hidden_dim, activation=C.sigmoid, input_rank=1) >> Label('gate')
        gate = dense(cont_attn) # [#, c][4*d]
        return gate*cont_attn

    def reasoning_layer(self, inputs, input_dim):
        input_ph = C.placeholder(shape=(input_dim,))
        rnn = create_birnn(GRU(self.hidden_dim), GRU(self.hidden_dim),'reasoning_gru')
        block = ResNetBlock(Sequential([
            LayerNormalization(name='layerbn'), Dropout(self.dropout), rnn
            ]))
        res = block(input_ph)
        return C.as_block(
            res,[(input_ph, inputs)], 'reasoning layer', 'reasoning layer'
        )
    
    def weighted_sum(self, inputs):
        input_ph = C.placeholder(shape=(self.2*hidden,))
        weight = Sequential([
            Dropout(self.dropout), Dense(self.hidden, activation=C.tanh),
            Dense(1,bias=False),
            C.softmax
        ])(input_ph) # [#,c][1]
        res = C.sequence.reduce_sum(weight*input_ph)
        return C.as_block(res,[(input_ph, inputs)], 'weighted sum','weighted sum')
    def output_layer(self, init, memory):

        def pointer(inputs, state):
            input_ph = C.placeholder(shape=(2*self.hidden_dim,))
            state_ph = C.placeholder(shape=(2*self.hidden_dim,))
            state_expand = C.sequence.broadcast_as(state_ph, input_ph)
            weight = Sequential([ C.splice, Dropout(self.dropout),
                Dense(self.hidden, activation=C.tanh),Dense(1,bias=False),
                C.softmax
            ])(input_ph, state_expand)
            res = C.sequence.reduce_sum(weight*input_ph)
            return C.as_block(
                C.combine(res, weight),
                [(input_ph, inputs), (state_ph, state)],
                'pointer', 'pointer')
        
        gru = GRU(self.hidden_dim)
        inp, logits1 = pointer(memory, init).outputs
        state2 = gru(init, inp)
        logits2 = pointer(memory, state2).outputs[1]

        return logits1, logits2 

    def build_model(self):
        q_axis = C.Axis.new_unique_dynamic_axis('query')
        c_axis = C.Axis.new_unique_dynamic_axis("context")
        b = C.Axis.default_batch_axis()
        # context 由于某些操作不支持稀疏矩阵，全部采用密集输入
        cgw = C.sequence.input_variable(self.wg_dim, sequence_axis=c_axis, name='cgw')
        cnw = C.sequence.input_variable(self.wn_dim, sequence_axis=c_axis, name='cnw')
        cc = C.input_variable(self.word_size, dynamic_axes = [b,c_axis], name='cc')
        # query
        qgw = C.sequence.input_variable(self.wg_dim, sequence_axis=q_axis, name='qgw')
        qnw = C.sequence.input_variable(self.wn_dim, sequence_axis=q_axis, name='qnw')
        qc = C.input_variable(self.word_size, dynamic_axes = [b,q_axis], name='qc')
        ab = C.sequence.input_variable(self.a_dim, sequence_axis=c_axis, name='ab')
        ae = C.sequence.input_variable(self.a_dim, sequence_axis=c_axis, name='ae')
        input_phs = {'cgw':cgw, 'cnw':cnw, 'qgw':qgw, 'qnw':qnw,
                        'cc':cc, 'qc':qc, 'ab':ab, 'ae':ae}

        self._input_phs = input_phs
        # graph
        qu, pu = self.input_layer(cgw, cnw, cc, qgw, qnw, qc)
        gate_pu = self.gate_attention_layer(pu, qu) # [#,c][4*hidden]
        pv = self.reasoning_layer(gate_pu, 4*self.hidden_dim) # [#,c][2*hidden]
        gate_self = self.gate_attention_layer(pv,pv) # [#,c][4*hidden]
        ph = self.reasoning_layer(gate_self, 4*self.hidden_dim) # [#,c][2*hidden]
        init_pu = self.weighted_sum(pu)
        start_logits, end_logits  = self.output_layer(init_pu, ph) # [#, c][1]
        
        # loss
        start_loss = seq_loss(start_logits, ab)
        end_loss = seq_loss(end_logits, ae)
        #paper_loss = start_loss + end_loss
        new_loss = all_spans_loss(start_logits, ab, end_logits, ae)
        self._model = c.combine([start_logits,end_logits])
        self._loss = new_loss
        return self._model, self._loss, self._input_phs
# =============== test edition ==================
from cntk.debugging import debug_model
def test_model_part():
    pass
def _testcode():
    data=[np.array([[1,2,3,0],[1,2,3,0]]),
        np.array([[1,2,0,0],[2,3,0,0]]),
        np.array([[4,0,0,0],[5,0,0,0],[6,0,0,0]])]
    inp=C.sequence.input_variable(4)

if __name__ == '__main__':
    create_rnet()
