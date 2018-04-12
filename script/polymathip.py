import polymath
from helpers import *

import cntk as C
from cntk.layers import *
import importlib

class BiDAFSL(polymath.BiDAF):
    def __init__(self, config_file):
        super(BiDAFSL, self).__init__(config_file)
        data_config = importlib.import_module(config_file).data_config
        model_config = importlib.import_module(config_file).model_config
        self.loss_lambda = model_config['loss_lambda']
        self._metric = None
    def attention_layer(self, context, query):
        input_ph = C.placeholder(shape=(2*self.hidden_dim,))
        input_mem = C.placeholder(shape=(2*self.hidden_dim,))
        with C.layers.default_options(bias=False, activation=C.relu):
            attn_proj_enc = C.layers.Dense(self.hidden_dim, init=glorot_uniform(), input_rank=1, name="Wqu")
            attn_proj_dec = C.layers.Dense(self.hidden_dim, init=glorot_uniform(), input_rank=1)

        inputs_ = attn_proj_enc(input_ph) # [#,c][d]
        memory_ = attn_proj_dec(input_mem) # [#,q][d]

        cln_mem_ph = C.placeholder() # [#,q][?]
        cln_inp_ph = C.placeholder() # [#,c][?]
        unpack_inputs, inputs_mask = C.sequence.unpack(cln_inp_ph, 0).outputs # [#][*=c,d] [#][*=c]
        matrix = C.times_transpose(cln_mem_ph, unpack_inputs)/(self.hidden_dim**0.5) # [#,q][*=c]
        expand_inputs = C.sequence.broadcast_as(unpack_inputs, cln_mem_ph) # [#,q][*=c,d]
        trans_expand_inputs = C.transpose(expand_inputs,[1,0]) # [#,q][d,*=c]
        q_over_c = C.reshape(C.reduce_sum(matrix*trans_expand_inputs,axis=1),(-1,))/(self.hidden_dim**0.5) # [#,q][d]

        weighted_c = q_over_c.clone(C.CloneMethod.share, {cln_mem_ph: memory_, cln_inp_ph: inputs_}) # [#,q][d]
        weighted_q = q_over_c.clone(C.CloneMethod.share, {cln_mem_ph: inputs_, cln_inp_ph: memory_}) # [#,c][d]
        c2c = seq_over_c.clone(C.CloneMethod.share, {cln_mem_ph: inputs_, cln_inp_ph: inputs_}) # [#,c][d]
        
        att_context = C.splice(input_ph, weighted_q, c2c)
        query_context = C.splice(input_mem, weighted_c)

        return C.as_block(
            C.combine(att_context, query_context),
            [(input_ph, context),(input_mem, query)],
            'attention_layer','attention_layer'
        )
    def encoder(self, doc1, doc2):
        doc1_ph = C.placeholder()
        doc2_ph = C.placeholder()
        proj_doc1 = Dense(self.hidden_dim)(doc1_ph)
        proj_doc2 = Dense(self.hidden_dim)(doc2_ph)
        birnn = OptimizedRnnStack(self.hidden_dim,bidirectional=True)
        summary1 = C.sequence.last(birnn(proj_doc1))
        summary2 = C.sequence.last(birnn(proj_doc2))
        return C.as_block(
            C.combine(summary1, summary2),
            [(doc1_ph, doc1),(doc2_ph, doc2)],
            'encoder', 'encoder'
        )

    def build_model(self):
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
        slc = C.input_variable(1, name='sl')
        input_phs = {'cgw':cgw, 'cnw':cnw, 'qgw':qgw, 'qnw':qnw,
                     'cc':cc, 'qc':qc, 'ab':ab, 'ae':ae, 'sl':slc}
        self._input_phs = input_phs

        #input layer
        cc = C.reshape(cc, (1,-1)); qc = C.reshape(qc, (1,-1))
        # 2*hidden_dim
        c_processed, q_processed = self.input_layer(cgw,cnw,cc,qgw,qnw,qc).outputs
        # 6*hidden 4*hidden
        att_context, att_query = self.attention_layer(c_processed, q_processed).outputs
        
        context_summary, query_summary= self.encoder(att_context, att_query).outputs
        cls_logits = Dense(1, activation=C.sigmoid)(C.splice(context_summary, query_summary))
        cls_loss = my_cross_entropy(cls_logits, slc)

        mod_context = self.modeling_layer(att_context)
        start_logits, end_logits = self.output_layer(mod_context)
        # span loss [#][1] + cls loss [#][1]
        new_loss = all_spans_loss(start_logits, ab, end_logits, ae)*slc + C.constant(self.loss_lambda)*cls_loss
        metric = C.classification_error(cls_logits, slc)
        res = C.combine([start_logits, end_logits, cls_logits])

        self._model = res
        self._loss = new_loss
        self._metric = metric
        return self._model, self._loss, self._input_phs
    @property
    def metric(self):
        if not self._metric:
            self.build_model()
        return self._metric

def my_cross_entropy(logits, labels):
    one = C.constant(1.0, name='one')
    loss = -C.constant(1.5)*labels*C.log(logits)-C.constant(0.5)*(one-labels)*C.log(one-logits)
    return loss
