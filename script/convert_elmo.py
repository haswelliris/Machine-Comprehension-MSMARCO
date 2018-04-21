import cntk as C
from cntk.layers import *
import numpy as np
import h5py
from helpers import HighwayBlock

class _ElmoCharEncoder(object):
    def __init__(self,weight_file='elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5'):
        self.weight_file= weight_file
        self.filter_num = 7
        self.highway_num = 2

    def _load_weight(self):
        self._load_char_embed()
        self._load_cnn_weight()
        self._load_highway()
        self._load_proj()
    def _load_char_embed(self):
        with h5py.File(self.weight_file, 'r') as f:
            tmp_weight = f['char_embed'][...] # shape: 261*16
        weight = np.zeros((tmp_weight.shape[0]+1, tmp_weight.shape[1]), dtype=np.float32)
        self.char_embed = C.constant(weight, name='elmo_char_embed')
    def _load_cnn_weight(self):
        self.convs = [None]*self.filter_num
        with h5py.File(self.weight_file,'r') as fin:
            for i in range(self.filter_num):
                weight = fin['CNN']['W_cnn_{}'.format(i)][...] # (1,h, w, out_c)
                bias = fin['CNN']['b_cnn_{}'.format(i)][...] # (int,)
                w_reshape = np.transpose(weight.squeeze(axis=0), axes=(2,0,1)) # (out_c, h, w) 

                self.convs[i] = Convolution2D((w_reshape.shape[1] ,w_reshape.shape[2]), w_reshape.shape[0],
                    init=w_reshape, reduction_rank=0, activation=C.relu,
                    init_bias=bias, name='char_conv_{}'.format(i))
    def _load_highway(self):
        self.highways = [None]*self.highway_num
        with h5py.File(self.weight_file,'r') as fin:
            for i in range(self.highway_num):
                w_transform = fin['CNN_high_{}'.format(i)]['W_transform'][...] # use for C.times(x,W)
                b_transform = fin['CNN_high_{}'.format(i)]['b_transform'][...]
                w_carry = fin['CNN_high_{}'.format(i)]['W_carry'][...] # use for (1-g)x+g*f(x)
                b_carry = fin['CNN_high_{}'.format(i)]['b_carry'][...]
                highways[i] = HighwayBlock(w_transform.shape[0],
                    transform_bias_initializer=w_transform,
                    transform_bias_initializer=b_transform,
                    update_weight_initializer=w_carry,
                    update_bias_initializer=b_carry)
    def _load_proj(self):
        with h5py.File(self.weight_file,'r') as fin:
            weight = fin['CNN_proj']['W_proj'][...]
            bias = fin['CNN_proj']['b_proj'][...]
            W_proj = C.contant(weight)
            b_proj = C.constant(bias)
        @C.Function
        def dense(x):
            return C.relu(C.times(x, W_proj)+b_proj)
        self.proj = dense
    def build(self, require_train=False):
        self._load_weight()
        @C.Function
        def _func(x):
            input_ph = C.placeholder()

            ph = C.placeholder()
            onehot_value = C.one_hot(ph,261)
            x1 = C.times(onehot_value, self.char_embed) # [#,*][50,16]
            # x2 = self.convs[0](x1) # [#,*][32,50,1]
            convs_res = []
            for i in range(self.filter_num):
                conv_res = self.convs[i](x1)
                x2 = C.reshape(C.reduce_max(conv_res, axis=1),(-1,))
            token_embed = C.splice(convs_res) # [#,*][2048]
            tmp_res = token_embed
            for i in range(self.highway_num):
                tmp_res = self.highways[i](tmp_res)
            highway_out=tmp_res # [#,*][2048]
            proj_out = self.proj(highway_out) # [#,*][512]

            if not require_train:
                res = proj_out.clone(C.CloneMethod.freeze, {ph:input_ph})
            else:
                res = proj_out.clone(C.CloneMethod.clone, {ph:input_ph})
            return C.as_block(
                res,[(input_ph, x)], 'elmo_char_encoder', 'elmo_char_encoder'
            )
    def test(self):
        input_ph=C.sequence.input_variable((50,))
        encoder = self.build()
        encode_out = encoder(input_ph)
        return encode_out
        
def proj_LSTM(input_dim, out_dim, init_W, init_H, init_b, init_W_0):
    '''numpy initial'''
    lstm = LSTM(4096)
    lstm.W = C.Parameter(shape=(input_dim, 4096*4),init=init_W) # (512,4096*4)
    lstm.H = C.Parameter(shape=(out_dim, 4*4096), init=init_H)
    lstm.b = C.Parameter(shape=(4096*4,), init=init_b)
    proj_W = C.Parameter(shape=(4096,out_dim), init=init_W_0)
    @C.Function
    def unit(dh, dc, x):
        ''' dh: out_dim, dc:4096, x:input_dim'''
        h, c = lstm(x) # h:4096 c:4096
        proj_h = C.times(h, proj_W) # out_dim
        return (proj_h, c) 
    return unit

class _ElmoBilm(object):
    def __inti__(self, weight_file='elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5')
        self.weight_file = weight_file
        self.layer_num = 2
        self.forward_unit = [LSTM(512, 4096) for _ in range(self.layer_num)]
        self.backward_unit = [LSTM(512, 4096) for _ in range(self.layer_num)]

    def _load_weight(self):
        with h5py.File(self.weight_file,'r') as fin:
            for i_layer, lstms in 
                enumerate(zip(self.forward_layers, self.backward_layers)):
                for j_direction, lstm in enumerate(lstms):
                    dataset = fin['RNN_%s' % j_direction]['RNN']['MultiRNNCell']['Cell%s' % i_layer]['LSTMCell']
                    # tensorflow packs the gates as input, memory, forget, output as same as cntk
                    tf_weights = np.transpose(dataset['W_0'][...]) # (16384, 1024)
                    tf_bias = dataset['B'][...] # (16384,)
                    # tensorflow adds 1.0 to forget gate bias instead of modifying the
                    # parameters...
                    tf_bias[4096*2:4096*3] += 1.0
                    proj_weights = np.transpose(dataset['W_P_0'][...]) # (4096, 512)

                    lstm.b = C.Parameter(16384, init=tf_bias)


if __name__=="__main__":
    pass