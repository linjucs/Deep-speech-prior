from __future__ import print_function
import tensorflow as tf
from tensorflow.contrib.layers import batch_norm, fully_connected, flatten
from tensorflow.contrib.layers import xavier_initializer
import numpy as np
import scipy.io.wavfile as wavfile


def leakyrelu(x, alpha=0.3, name='lrelu'):
    return tf.maximum(x, alpha * x, name=name)

def deconv(x, output_shape, kwidth=5, dilation=2, init=None, uniform=False,
           bias_init=None, name='deconv1d'):
    input_shape = x.get_shape()
    in_channels = input_shape[-1]
    out_channels = output_shape[-1]
    assert len(input_shape) >= 3
    # reshape the tensor to use 2d operators
    x2d = tf.expand_dims(x, 2)
    o2d = output_shape[:2] + [1] + [output_shape[-1]]
    w_init = init
    if w_init is None:
        w_init = xavier_initializer(uniform=uniform)
    with tf.variable_scope(name):
        # filter shape: [kwidth, output_channels, in_channels]
        W = tf.get_variable('W', [kwidth, 1, out_channels, in_channels],
                            initializer=w_init
                            )
        try:
            deconv = tf.nn.conv2d_transpose(x2d, W, output_shape=o2d,
                                            strides=[1, dilation, 1, 1])
        except AttributeError:
            # support for versions of TF before 0.7.0
            # based on https://github.com/carpedm20/DCGAN-tensorflow
            deconv = tf.nn.deconv2d(x2d, W, output_shape=o2d,
                                    strides=[1, dilation, 1, 1])
        if bias_init is not None:
            b = tf.get_variable('b', [out_channels],
                                initializer=tf.constant_initializer(0.))
            deconv = tf.reshape(tf.nn.bias_add(deconv, b), deconv.get_shape())
        else:
            deconv = tf.reshape(deconv, deconv.get_shape())
        # reshape back to 1d
        deconv = tf.reshape(deconv, output_shape)
        return deconv

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

def read_audio(filename):
    fm, wav_data = wavfile.read(filename)
    if fm != 16000:
        raise ValueError('Sampling rate is expected to be 16kHz!')
    return wav_data[:32768]

def make_noise(x, mean=0., std=1., name='z'):
    z = tf.random_normal(x.shape, mean=mean, stddev=std,
                                     name=name, dtype=tf.float32)
    return z
    
class AEGenerator(object):

    def __init__(self):
        self.g_dilated_blocks = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
        # num fmaps foro AutoEncoder SEGAN (v1)
        self.bias_deconv = False
        self.deconv_type = 'deconv'
        self.g_enc_depths = [16, 32, 32, 64, 64, 128, 128, 256, 256, 512, 1024]
        # Define D fmaps
        self.d_num_fmaps = [16, 32, 32, 64, 64, 128, 128, 256, 256, 512, 1024]

    def __call__(self, noisy_w, is_ref, spk=None, z_on=True, do_prelu=False):
        # TODO: remove c_vec
        """ Build the graph propagating (noisy_w) --> x
        On first pass will make variables.
        """

        def make_z(shape, mean=0., std=1., name='z'):
            if is_ref:
                with tf.variable_scope(name) as scope:
                    z_init = tf.random_normal_initializer(mean=mean, stddev=std)
                    z = tf.get_variable("z", shape,
                                        initializer=z_init,
                                        trainable=False
                                        )
            else:
                z = tf.random_normal(shape, mean=mean, stddev=std,
                                     name=name, dtype=tf.float32)
            return z

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
            for layer_idx, layer_depth in enumerate(self.g_enc_depths):
                bias_init = None
                h_i_dwn = downconv(h_i, layer_depth, kwidth=kwidth,
                                   init=tf.truncated_normal_initializer(stddev=0.02),
                                   bias_init=bias_init,
                                   name='enc_{}'.format(layer_idx))
                if is_ref:
                    print('Downconv {} -> {}'.format(h_i.get_shape(),
                                                     h_i_dwn.get_shape()))
                h_i = h_i_dwn
                if layer_idx < len(self.g_enc_depths) - 1:
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
                z = make_z([1, h_i.get_shape().as_list()[1],
                            self.g_enc_depths[-1]])
                h_i = tf.concat([z, h_i], 2)

            #SECOND DECODER (reverse order)
            g_dec_depths = self.g_enc_depths[:-1][::-1] + [1]
            if is_ref:
                print('g_dec_depths: ', g_dec_depths)
            for layer_idx, layer_depth in enumerate(g_dec_depths):
                h_i_dim = h_i.get_shape().as_list()
                out_shape = [h_i_dim[0], h_i_dim[1] * 2, layer_depth]
                bias_init = None
                # deconv
                if self.deconv_type == 'deconv':
                    if is_ref:
                        print('-- Transposed deconvolution type --')
                        if self.bias_deconv:
                            print('Biasing deconv in G')
                    h_i_dcv = deconv(h_i, out_shape, kwidth=kwidth, dilation=2,
                                     init=tf.truncated_normal_initializer(stddev=0.02),
                                     bias_init=bias_init,
                                     name='dec_{}'.format(layer_idx))
                elif self.deconv_type == 'nn_deconv':
                    if is_ref:
                        print('-- NN interpolated deconvolution type --')
                        if self.bias_deconv:
                            print('Biasing deconv in G')
                    if self.bias_deconv:
                        bias_init = 0.
                    h_i_dcv = nn_deconv(h_i, kwidth=kwidth, dilation=2,
                                        init=tf.truncated_normal_initializer(stddev=0.02),
                                        bias_init=bias_init,
                                        name='dec_{}'.format(layer_idx))
                else:
                    raise ValueError('Unknown deconv type {}'.format(self.deconv_type))
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
                    h_i = tf.concat ([h_i, skip_], 2)

                else:
                    if is_ref:
                        print('-- Dec: tanh activation --')
                    h_i = tf.tanh(h_i)

            wave = h_i
            if is_ref and do_prelu:
                print('Amount of alpha vectors: ', len(alphas))
            if is_ref:
                print('Amount of skip connections: ', len(skips))
                print('Last wave shape: ', wave.get_shape())
                print('*************************')
            self.generator_built = True
            # ret feats contains the features refs to be returned
            ret_feats = [wave]
            if z_on:
                ret_feats.append(z)
            if is_ref and do_prelu:
                ret_feats += alphas
            return ret_feats

if __name__=='__main__':
    wav_path = 'test.wav'
    epoch = 60
    audio = np.squeeze(read_audio(wav_path)) # need to rescale, [-1, 1] or [0, 1] ???
    signal = audio / np.max(np.abs(audio), axis=0)
    signal = np.expand_dims(signal, 0)
    noise = np.random.uniform(-1.0, 1.0, size=[1,32768]) # need to change if rescale to [0, 1]
    learning_rate = 0.0002
    save_frequency = 2

    noise_in = tf.placeholder('float', [1, 32768])
    signal_out = tf.placeholder('float', [1, 32768])
    ae = AEGenerator()
    G, z  = ae(noise_in, is_ref=True, spk=None, do_prelu=False)
    loss = tf.reduce_sum(tf.abs(tf.subtract(tf.squeeze(G,-1),
                               signal_out)))
    optimizer = tf.train.AdagradOptimizer(learning_rate).minimize(loss)
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    for iteration in range(epoch):
        _, curr_loss = sess.run([optimizer, loss], 
                    feed_dict={noise_in: noise, signal_out: signal})
        if iteration % save_frequency == 0:
            print('Epoch', iteration, '/', epoch, 'loss:',curr_loss)
