import cntk as C
from cntk.layers import *
from helpers import *
import polymath

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
        with C.layers.default_options(bias=False, activation=C.relu): # all the projections have no bias
            attn_proj_enc = C.layers.Dense(self.hidden_dim, init=glorot_uniform(), input_rank=1, name="Wqu")
            attn_proj_dec = C.layers.Dense(self.hidden_dim, init=glorot_uniform(), input_rank=1)

        inputs_ = attn_proj_enc(inputs) # [#,c][d]
        memory_ = attn_proj_dec(memory) # [#,q][d]
        outputs = C.times_transpose(inputs_, memory_)/(self.hidden_dim**0.5)
        logits = C.softmax(outputs)
        weighted_att = C.times(logits, memory)


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
        qu, pu = input_layer(cgw, cnw, cc, qgw, qnw, qc)

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
# =============== test edition ==================
from cntk.debugging import debug_model
def test_model_part():
    q_axis = C.Axis.new_unique_dynamic_axis('query')
    c_axis = C.Axis.new_unique_dynamic_axis("context")
    b = C.Axis.default_batch_axis()
    # context
    cgw = C.sequence.input_variable(wg_dim, sequence_axis=c_axis, is_sparse=False, name='cgw')
    cnw = C.sequence.input_variable(wn_dim, sequence_axis=c_axis, is_sparse=False, name='cnw')
    cc = C.input_variable(word_size, dynamic_axes = [b,c_axis], name='cc')
    # query
    qgw = C.sequence.input_variable(wg_dim, sequence_axis=q_axis, is_sparse=False, name='qgw')
    qnw = C.sequence.input_variable(wn_dim, sequence_axis=q_axis, is_sparse=False, name='qnw')
    qc = C.input_variable(word_size, dynamic_axes = [b,q_axis], name='qc')
    ab = C.sequence.input_variable(1, sequence_axis=c_axis, name='ab')
    ae = C.sequence.input_variable(1, sequence_axis=c_axis, name='ae')

    # graph
    qu, pu = input_layer(cgw, cnw, cc, qgw, qnw, qc)
    pv = gate_attention_recurrence_layer(qu, pu)
    ph = self_match_attention(pv)
    Wqu = pv.find_by_name('Wqu').parameters[0]
    rq = attention_pooling_layer(Wqu, qu)
    return C.combine(rq,ab,ae)
def _testcode():
    data=[np.array([[1,2,3,0],[1,2,3,0]]),
        np.array([[1,2,0,0],[2,3,0,0]]),
        np.array([[4,0,0,0],[5,0,0,0],[6,0,0,0]])]
    inp=C.sequence.input_variable(4)

if __name__ == '__main__':
    create_rnet()
