import polymath

import cntk as C
from cntk.layers import *

class BiDAFSL(polymath.BiDAF):
    def __init__(self, config_file):
        super(BiDAFSL, self).__init__(config_file)
        data_config = importlib.import_module(config_file).data_config
        model_config = importlib.import_module(config_file).model_config
        self.loss_lambda = model_config['loss_lambda']
    def attention_layer(self, context, query):
        input_ph = C.placeholder(shape=(2*self.hidden_dim,))
        input_mem = C.placeholder(shape=(2*self.hidden_dim,))
        with C.layers.default_options(bias=False, activation=C.relu):
            attn_proj_enc = C.layers.Dense(self.hidden_dim, init=glorot_uniform(), input_rank=1, name="Wqu")
            attn_proj_dec = C.layers.Dense(self.hidden_dim, init=glorot_uniform(), input_rank=1)

        inputs_ = attn_proj_enc(input_ph) # [#,c][d]
        memory_ = attn_proj_dec(input_mem) # [#,q][d]

        cln_mem_ph = C.placeholder(shape=(self.hidden_dim, ))
        cln_inp_ph = C.placeholder(shape=(self.hidden_dim, ))
        unpack_memory, mem_mask = C.sequence.unpack(cln_mem_ph, 0).outputs # [#][*=q, d], [#][*=q]
        unpack_inputs, inputs_mask = C.sequence.unpack(cln_inp_ph, 0).outputs # [#][*=c,d] [#][*=c]
        matrix = C.times_transpose(unpack_inputs, unpack_memory)/(self.hidden_dim**0.5) # [#][*=c,*=q]
        over_q = C.transpose(C.times(C.transpose(unpack_inputs, matrix))) # [#][*=q,d]
        over_c = C.times_transpose(matrix ,C.splice(unpack_memory, over_q, axis = 1)) # [#][*=c, 2d]
        seq_over_c = C.sequence.gather(over_c, inputs_mask, cln_inp_ph.dymamic_axis) # [#, c][2*d]

        seq_over_c.clone(C.CloneMethod.clone, {})



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
        c_processed, q_processed = self.input_layer(cgw,cnw,cc,qgw,qnw,qc).outputs

