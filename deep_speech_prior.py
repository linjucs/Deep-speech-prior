from __future__ import print_function
import tensorflow as tf
from tensorflow.contrib.layers import batch_norm, fully_connected, flatten
from tensorflow.contrib.layers import xavier_initializer
import numpy as np

def downconv(x, output_dim, kwidth=5, pool=2, init=None, uniform=False,
             bias_init=None, name='downconv'):
    """ Downsampled convolution 1d """
    x2d = tf.expand_dims(x, 2)
    w_init = init
    if w_init is None:
        w_init = xavier_initializer(uniform=uniform)
    with tf.variable_scope(name):
        W = tf.get_variable('W', [kwidth, 1, x.get_shape()[-1], output_dim],
                            initializer=w_init)
        conv = tf.nn.conv2d(x2d, W, strides=[1, pool, 1, 1], padding='SAME')
        if bias_init is not None:
            b = tf.get_variable('b', [output_dim],
                                initializer=bias_init)
            conv = tf.reshape(tf.nn.bias_add(conv, b), conv.get_shape())
        else:
            conv = tf.reshape(conv, conv.get_shape())
        # reshape back to 1d
        conv = tf.reshape(conv, conv.get_shape().as_list()[:2] +
                          [conv.get_shape().as_list()[-1]])
        return conv

class AEGenerator(object):

    def __init__(self, segan):
        self.segan = segan

    def __call__(self, noisy_w, is_ref, spk=None, z_on=True, do_prelu=False):
        # TODO: remove c_vec
        """ Build the graph propagating (noisy_w) --> x
        On first pass will make variables.
        """
        segan = self.segan

        def make_z(shape, mean=0., std=1., name='z'):
            if is_ref:
                with tf.variable_scope(name) as scope:
                    z_init = tf.random_normal_initializer(mean=mean, stddev=std)
                    z = tf.get_variable("z", shape,
                                        initializer=z_init,
                                        trainable=False
                                        )
                    if z.device != "/device:GPU:0":
                        # this has to be created into gpu0
                        print('z.device is {}'.format(z.device))
                        assert False
            else:
                z = tf.random_normal(shape, mean=mean, stddev=std,
                                     name=name, dtype=tf.float32)
            return z

        if hasattr(segan, 'generator_built'):
            tf.get_variable_scope().reuse_variables()
            make_vars = False
        else:
            make_vars = True
        if is_ref:
            print('*** Building Generator ***')
        in_dims = noisy_w.get_shape().as_list()
        h_i = noisy_w
        if len(in_dims) == 2:
            h_i = tf.expand_dims(noisy_w, -1)
        elif len(in_dims) < 2 or len(in_dims) > 3:
            raise ValueError('Generator input must be 2-D or 3-D')
        kwidth = 31
        enc_layers = 7
        skips = []
        if is_ref and do_prelu:
            #keep track of prelu activations
            alphas = []
        with tf.variable_scope('g_ae'):
            #AE to be built is shaped:
            # enc ~ [16384x1, 8192x16, 4096x32, 2048x32, 1024x64, 512x64, 256x128, 128x128, 64x256, 32x256, 16x512, 8x1024]
            # dec ~ [8x2048, 16x1024, 32x512, 64x512, 8x256, 256x256, 512x128, 1024x128, 2048x64, 4096x64, 8192x32, 16384x1]
            #FIRST ENCODER
            for layer_idx, layer_depth in enumerate(segan.g_enc_depths):
                bias_init = None
                if segan.bias_downconv:
                    if is_ref:
                        print('Biasing downconv in G')
                    bias_init = tf.constant_initializer(0.)
                h_i_dwn = downconv(h_i, layer_depth, kwidth=kwidth,
                                   init=tf.truncated_normal_initializer(stddev=0.02),
                                   bias_init=bias_init,
                                   name='enc_{}'.format(layer_idx))
                if is_ref:
                    print('Downconv {} -> {}'.format(h_i.get_shape(),
                                                     h_i_dwn.get_shape()))
                h_i = h_i_dwn
                if layer_idx < len(segan.g_enc_depths) - 1:
                    if is_ref:
                        print('Adding skip connection downconv '
                              '{}'.format(layer_idx))
                    # store skip connection
                    # last one is not stored cause it's the code
                    skips.append(h_i)
                if do_prelu:
                    if is_ref:
                        print('-- Enc: prelu activation --')
                    h_i = prelu(h_i, ref=is_ref, name='enc_prelu_{}'.format(layer_idx))
                    if is_ref:
                        # split h_i into its components
                        alpha_i = h_i[1]
                        h_i = h_i[0]
                        alphas.append(alpha_i)
                else:
                    if is_ref:
                        print('-- Enc: leakyrelu activation --')
                    h_i = leakyrelu(h_i)

            if z_on:
                # random code is fused with intermediate representation
                z = make_z([segan.batch_size, h_i.get_shape().as_list()[1],
                            segan.g_enc_depths[-1]])
                h_i = tf.concat(2, [z, h_i])

            #SECOND DECODER (reverse order)
            g_dec_depths = segan.g_enc_depths[:-1][::-1] + [1]
            if is_ref:
                print('g_dec_depths: ', g_dec_depths)
            for layer_idx, layer_depth in enumerate(g_dec_depths):
                h_i_dim = h_i.get_shape().as_list()
                out_shape = [h_i_dim[0], h_i_dim[1] * 2, layer_depth]
                bias_init = None
                # deconv
                if segan.deconv_type == 'deconv':
                    if is_ref:
                        print('-- Transposed deconvolution type --')
                        if segan.bias_deconv:
                            print('Biasing deconv in G')
                    if segan.bias_deconv:
                        bias_init = tf.constant_initializer(0.)
                    h_i_dcv = deconv(h_i, out_shape, kwidth=kwidth, dilation=2,
                                     init=tf.truncated_normal_initializer(stddev=0.02),
                                     bias_init=bias_init,
                                     name='dec_{}'.format(layer_idx))
                elif segan.deconv_type == 'nn_deconv':
                    if is_ref:
                        print('-- NN interpolated deconvolution type --')
                        if segan.bias_deconv:
                            print('Biasing deconv in G')
                    if segan.bias_deconv:
                        bias_init = 0.
                    h_i_dcv = nn_deconv(h_i, kwidth=kwidth, dilation=2,
                                        init=tf.truncated_normal_initializer(stddev=0.02),
                                        bias_init=bias_init,
                                        name='dec_{}'.format(layer_idx))
                else:
                    raise ValueError('Unknown deconv type {}'.format(segan.deconv_type))
                if is_ref:
                    print('Deconv {} -> {}'.format(h_i.get_shape(),
                                                   h_i_dcv.get_shape()))
                h_i = h_i_dcv
                if layer_idx < len(g_dec_depths) - 1:
                    if do_prelu:
                        if is_ref:
                            print('-- Dec: prelu activation --')
                        h_i = prelu(h_i, ref=is_ref,
                                    name='dec_prelu_{}'.format(layer_idx))
                        if is_ref:
                            # split h_i into its components
                            alpha_i = h_i[1]
                            h_i = h_i[0]
                            alphas.append(alpha_i)
                    else:
                        if is_ref:
                            print('-- Dec: leakyrelu activation --')
                        h_i = leakyrelu(h_i)
                    # fuse skip connection
                    skip_ = skips[-(layer_idx + 1)]
                    if is_ref:
                        print('Fusing skip connection of '
                              'shape {}'.format(skip_.get_shape()))
                    h_i = tf.concat(2, [h_i, skip_])

                else:
                    if is_ref:
                        print('-- Dec: tanh activation --')
                    h_i = tf.tanh(h_i)

            wave = h_i
            if is_ref and do_prelu:
                print('Amount of alpha vectors: ', len(alphas))
            segan.gen_wave_summ = histogram_summary('gen_wave', wave)
            if is_ref:
                print('Amount of skip connections: ', len(skips))
                print('Last wave shape: ', wave.get_shape())
                print('*************************')
            segan.generator_built = True
            # ret feats contains the features refs to be returned
            ret_feats = [wave]
            if z_on:
                ret_feats.append(z)
            if is_ref and do_prelu:
                ret_feats += alphas
            return ret_feats

