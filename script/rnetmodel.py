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
def create_attention(attention_dim, name=""):
    # model parameters
    with C.layers.default_options(bias=False): # all the projections have no bias
        attn_proj_enc   = C.layers.Dense(attention_dim, init=glorot_uniform(), input_rank=1, name="Wqu")
        attn_proj_dec   = C.layers.Dense(attention_dim, init=glorot_uniform(), input_rank=1, name="attn_proj_dec")
        attn_proj_tanh  = C.layers.Dense(1, init=glorot_uniform(), input_rank=1, name="attn_proj_tanh")

    @C.Function
    def new_attention(encoder_hidden_state, decoder_hidden_state):
        # encode_hidden_state: [#, e] [h]
        # decoder_hidden_state: [#, d] [H]
        unpacked_encoder_hidden_state, valid_mask = C.sequence.unpack(encoder_hidden_state, padding_value=0).outputs
        # unpacked_encoder_hidden_state: [#] [*=e, h]
        # valid_mask: [#] [*=e]
        projected_encoder_hidden_state = C.sequence.broadcast_as(attn_proj_enc(unpacked_encoder_hidden_state), decoder_hidden_state)
        # projected_encoder_hidden_state: [#, d] [*=e, attention_dim]
        broadcast_valid_mask = C.sequence.broadcast_as(C.reshape(valid_mask, (1,), 1), decoder_hidden_state)
        # broadcast_valid_mask: [#, d] [*=e]
        projected_decoder_hidden_state = attn_proj_dec(decoder_hidden_state)
        # projected_decoder_hidden_state: [#, d] [attention_dim]
        tanh_output = C.tanh(projected_decoder_hidden_state + projected_encoder_hidden_state)
        # tanh_output: [#, d] [*=e, attention_dim]
        attention_logits = attn_proj_tanh(tanh_output)
        # attention_logits = [#, d] [*=e, 1]
        minus_inf = C.constant(-1e+30)
        masked_attention_logits = C.element_select(broadcast_valid_mask, attention_logits, minus_inf)
        # masked_attention_logits = [#, d] [*=e]
        attention_weights = C.softmax(masked_attention_logits, axis=0)
        attention_weights = C.layers.Label('attention_weights')(attention_weights)
        # attention_weights = [#, d] [*=e]
        attended_encoder_hidden_state = C.reduce_sum(attention_weights * C.sequence.broadcast_as(unpacked_encoder_hidden_state, attention_weights), axis=0)
        # attended_encoder_hidden_state = [#, d] [1, h]
        output = C.reshape(attended_encoder_hidden_state, (), 0, 1)
        # output = [#, d], [h]
        return output
    return new_attention
def Attention3State(attention_dim, name=""):
    attn = create_attention(attention_dim, name=name)
    def AttentionAdapter(encoder_hidden_state, decoder_hidden_state, related_state):
        decoder_hidden = C.splice(decoder_hidden_state,related_state)
        return attn(encoder_hidden_state, decoder_hidden)
    return AttentionAdapter
# =============== layer function ================
def attention_layer(qu, pu, decoder_hidden_state):
    with C.layers.default_options(bias=False): # all the projections have no bias
        attn_proj_enc = C.layers.Dense(hidden_dim, init=glorot_uniform(), input_rank=1, name="Wqu")
        attn_proj_rel = C.layers.Dense(hidden_dim, init=glorot_uniform(), input_rank=1, name="attn_proj_rel")
        attn_proj_dec = C.layers.Dense(hidden_dim, init=glorot_uniform(), input_rank=1)
        attn_proj_tanh  = C.layers.Dense(1, init=glorot_uniform(), input_rank=1, name="attn_proj_tanh")
    # preprocess
    unpacked_qu, valid_mask = C.sequence.unpack(qu, padding_value=0).outputs # 填充让每个问题长度相同
    projected_encoder_hidden_state = C.sequence.broadcast_as(attn_proj_enc(unpacked_qu), pu) # 对passage中的每个词都配置一份问题向量
    broadcast_valid_mask = C.sequence.broadcast_as(C.reshape(valid_mask, (1,), 1), pu)
    projected_decoder_hidden_state = attn_proj_rel(pu)
    processed = projected_encoder_hidden_state+projected_decoder_hidden_state

    tanh_output = C.tanh(processed+attn_proj_dec(decoder_hidden_state)) # 每个p的单词，都有len(q)份tanh输出
    attention_logits = attn_proj_tanh(tanh_output) # tanh值向量映射为标量
    minus_inf = C.constant(-1e+30)
    masked_attention_logits = C.element_select(broadcast_valid_mask, attention_logits, minus_inf)
    # masked_attention_logits = [#, d] [*=e]
    attention_weights = C.softmax(masked_attention_logits, axis=0) # 每个p的单词，都有len(q)份权重
    attention_weights = C.layers.Label('attention_weights')(attention_weights)
    # 每个p的单词，都有1份attention向量
    attended_encoder_hidden_state = C.reduce_sum(attention_weights * C.sequence.broadcast_as(unpacked_qu, attention_weights), axis=0)
    output = C.reshape(attended_encoder_hidden_state, (), 0, 1) # 去掉len=1的一层
    return output
def gate_attention_recurrence_layer(qu, pu):

    # parameter
    r = C.layers.Recurrence(C.layers.RNNStep(hidden_dim))
    # graph
    hidden_state_ph = C.layers.ForwardDeclaration(name='hidden_state_ph')
    attention_context = attention_layer(qu, pu, hidden_state_ph) # 每一个p中的单词，都有相应的attention向量
    attention_context = C.reconcile_dynamic_axes(attention_context, pu)
    raw_state = C.splice(pu, attention_context)
    gate = C.sigmoid(C.layers.Dense(2*hidden_dim, bias=False, init=glorot_uniform(), input_rank=1)(raw_state))
    rnn_input = C.element_times(gate,raw_state)
    rnn_output = r(rnn_input)
    prevout=C.sequence.past_value(rnn_output)
    hidden_state_ph.resolve_to(prevout)
    print('gate rnn input shape:{}'.format(rnn_input.output))
    print('gate rnn output shape:{}'.format(rnn_output.output))
    return rnn_output

    # runit = C.layers.RNNStep(hidden_dim)
    # attn = create_attention(hidden_dim)

    # @C.Function
    # def model(qu_ph,pu_ph):
    #     @C.Function
    #     def gateUnit(prev, nowinp):
    #         nowinp=C.reconcile_dynamic_axes(nowinp, prev)
    #         attin = C.splice(prev, nowinp)
    #         attout=attn(qu_ph, attin)
    #         attout=C.reconcile_dynamic_axes(attout, nowinp)
    #         rnnin=C.splice(attout, nowinp)
    #         rnnout=runit(prev, rnnin)
    #         return rnnout
    #     rec = C.layers.Recurrence(gateUnit)
    #     res = rec(pu_ph)
    #     return res

    # pv = model(qu, pu)
    # return pv
def self_match_attention(pv):
    # parameter
    attention_model = C.layers.AttentionModel(hidden_dim)
    with C.layers.default_options(enable_self_stabilization=True):
        birnn_ph = create_birnn(C.layers.RNNStep(hidden_dim//2), C.layers.RNNStep(hidden_dim//2))

    # graph
    attention_context = attention_model(pv, pv)
    attention_context = C.reconcile_dynamic_axes(attention_context, pv)
    rnn_input = C.splice(pv, attention_context)
    ph = birnn_ph(rnn_input)
    print('self match input shape:{}'.format(rnn_input.output))
    print('self match output shape:{}'.format(ph.output))
    return ph
def attention_pooling_layer(weight, qu):
    with C.layers.default_options(bias=False):
        V = C.parameter(hidden_dim)
        attn_proj_tanh = C.layers.Dense(1, init=glorot_uniform(), input_rank=1, name="attn_proj_tanh")
    tanh_output = C.tanh(C.times(qu, weight)+V)
    proj_tanh_output = attn_proj_tanh(tanh_output)
    attention_weights = C.softmax(proj_tanh_output)
    rq = C.sequence.reduce_sum(attention_weights*qu)
    print('weight shape:{}'.format(weight))
    print('tanh output shape:{}'.format(tanh_output.output))
    print('attention weight shape:{}'.format(attention_weights.output))
    print('rq shape:{}'.format(rq.output))
    return rq
def output_layer(init_state, input_state):
    # parameter
    r = C.layers.RNNStep(hidden_dim)
    attention_model=C.layers.AttentionModel(hidden_dim)
    # graph
    attention_context = attention_model(input_state, init_state)
    h1 = r(init_state, attention_context)
    attention_weights_start = attention_context.find_all_with_name('attention_weights')[0].output
    start_pos = C.argmax(attention_weights_start, axis=0)
    attention_context = attention_model(input_state, h1)
    attention_weights_end = attention_context.find_all_with_name('attention_weights')[0].output
    end_pos = C.argmax(attention_weights_end, axis=0)

    attention_weights_start = C.to_sequence(attention_weights_start)
    attention_weights_end = C.to_sequence_like(attention_weights_end, attention_weights_start)
    print('output layer rnn output shape:{}'.format(h1.output))
    print('output attention context shape:{}'.format(attention_context.output))
    print('output attention weight shape:{}'.format(attention_weights_end.output))
    print('pointer shape:{}'.format(start_pos.output))
    return attention_weights_start, attention_weights_end
def create_rnet():

    pv = gate_attention_recurrence_layer(qu, pu)
    ph = self_match_attention(pv)
    Wqu = pv.find_by_name('Wqu').parameters[0]
    rq = attention_pooling_layer(Wqu, qu)
    start_pos, end_pos = output_layer(rq, ph)

    # loss
    start_loss = seq_loss(start_pos, ab)
    end_loss = seq_loss(end_pos, ae)
    #paper_loss = start_loss + end_loss
    new_loss = all_spans_loss(start_pos, ab, end_pos, ae)
    return C.combine([start_pos, end_pos]), new_loss

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
