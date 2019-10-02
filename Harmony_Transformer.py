import numpy as np # version 1.14.5
import random
import tensorflow as tf # version 1.11
from tensorflow.python.framework import ops

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class Harmony_Transformer(object):
    def __init__(self,
                 frequency_size=24,
                 segment_width=21,
                 n_steps=100,
                 n_classes=26,
                 encoder_input_embedding_size=512,
                 decoder_input_embedding_size=512,
                 initial_learning_rate=1e-4,
                 dropout_rate=0.5,
                 annealing_rate=1.1,
                 batch_size=60,
                 lambda_loss_ct=3,
                 lambda_loss_c=1,
                 lambda_L2=2e-4,
                 training_steps=100000):
        self._frequency_size = frequency_size
        self._segment_width = segment_width
        self._feature_size = frequency_size * segment_width # input size (feature size)
        self._n_steps = n_steps
        self._n_classes= n_classes
        self._encoder_input_embedding_size = encoder_input_embedding_size
        self._decoder_input_embedding_size = decoder_input_embedding_size
        self._session = None
        self._graph = None
        self._lambda_loss_ct = lambda_loss_ct
        self._lambda_loss_c = lambda_loss_c,
        self._lambda_L2 = lambda_L2
        self._dropout_rate = dropout_rate
        self._annealing_rate = annealing_rate
        self._batch_size = batch_size
        self._initial_learning_rate = initial_learning_rate
        self._training_steps = training_steps

    def _normalize(self, inputs, epsilon=1e-8, scope="ln", reuse=None):
        '''Applies layer normalization.'''
        with tf.variable_scope(scope, reuse=reuse):
            inputs_shape = inputs.get_shape()
            params_shape = inputs_shape[-1:]
            mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
            beta = tf.get_variable("beta", params_shape, initializer=tf.zeros_initializer())
            gamma = tf.get_variable("gamma", params_shape, initializer=tf.ones_initializer())
            normalized = (inputs - mean) / ((variance + epsilon) ** (.5))
            outputs = gamma * normalized + beta

        return outputs

    def _feedforward(self, inputs, n_units=[2048, 512], activation_function=tf.nn.relu, dropout_rate=0, is_training=True, scope="feedforward", reuse=None):
        '''Point-wise feed forward net.'''
        with tf.variable_scope(scope, reuse=reuse):
            # Inner layer
            params = {"inputs": inputs, "filters": n_units[0], "kernel_size": 1, "activation": activation_function, "use_bias": True}
            outputs = tf.layers.conv1d(**params)

            # Readout layer
            params = {"inputs": outputs, "filters": n_units[1], "kernel_size": 1, "activation": None, "use_bias": True}
            outputs = tf.layers.conv1d(**params)

            # Dropout
            outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=is_training)

            # Residual connection
            outputs += inputs

            # Normalize
            outputs = self._normalize(outputs)

        return outputs

    def _positional_encoding(self, batch_size, time_steps, n_units, zero_pad=False, scale=False, scope="positional_encoding", reuse=None):
        '''Sinusoidal Positional_Encoding.'''

        N, T = batch_size, time_steps
        with tf.variable_scope(scope, reuse=reuse):
            position_ind = tf.tile(tf.expand_dims(tf.range(T), 0), [N, 1])

            # First part of the PE function: sin and cos argument
            position_enc = np.array([[pos / np.power(10000, 2.*i/n_units) for i in range(n_units)] for pos in range(T)], dtype=np.float32)

            # Second part, apply the cosine to even columns and sin to odds.
            position_enc[:, 0::2] = np.sin(position_enc[:, 0::2]) # dim 2i
            position_enc[:, 1::2] = np.cos(position_enc[:, 1::2]) # dim 2i+1

            # Convert to a tensor
            lookup_table = tf.convert_to_tensor(position_enc)

            if zero_pad:
                lookup_table = tf.concat((tf.zeros(shape=[1, n_units]), lookup_table[1:, :]), 0)
            outputs = tf.nn.embedding_lookup(lookup_table, position_ind)

            if scale:
                outputs = outputs * n_units**0.5

        return outputs

    def _get_relative_position(self, n_steps, n_units=512, max_dist=2, name='relative_position_encodings'):

        n_vectors = 2 * max_dist + 1
        center = n_vectors // 2
        pos_enc = tf.get_variable(name, dtype=tf.float32, shape=[n_vectors, n_units], initializer=tf.contrib.layers.xavier_initializer())

        n_left = [min(max_dist, i) for i in range(n_steps)]
        n_right = n_left[::-1]
        pos_enc_pad = []
        self = tf.expand_dims(pos_enc[center], 0)
        for i, n_l, n_r in zip(range(n_steps), n_left, n_right):
            left = pos_enc[(center - n_l):center]
            right = pos_enc[(center + 1):(center + 1 + n_r)]
            temp = tf.concat([left, self, right], axis=0)

            n_left_pad = i - n_l
            n_right_pad = n_steps - i - n_r - 1
            if n_left_pad > 0:
                temp = tf.concat([tf.reshape(tf.tile(temp[0], [n_left_pad]), [n_left_pad, n_units]), temp], axis=0)
            if n_right_pad > 0:
                temp = tf.concat([temp, tf.reshape(tf.tile(temp[-1], [n_right_pad]), [n_right_pad, n_units])], axis=0)

            pos_enc_pad.append(temp)

        return tf.stack(pos_enc_pad) # [n_steps, n_steps, n_units]

    def _multihead_attention(self, queries, keys, values=None, n_units=None, n_heads=8, activation_function=tf.nn.relu,
                             causal=False, relative_position=False, max_dist=16, self_mask=False,
                             dropout_rate=0, is_training=True, scope="multihead_attention", reuse=None):
        '''Applies multihead attention.'''
        with tf.variable_scope(scope, reuse=reuse):
            # Set the fall back option for n_units
            if values is None:
                values = keys

            if n_units is None:
                n_units = queries.get_shape().as_list[-1]

            # Linear projections
            Q = tf.layers.dense(queries, n_units, activation=activation_function)  # (N, T_q, C)
            K = tf.layers.dense(keys, n_units, activation=activation_function)  # (N, T_k, C)
            V = tf.layers.dense(values, n_units, activation=activation_function)  # (N, T_k, C)

            # Split and concat
            Q_ = tf.concat(tf.split(Q, n_heads, axis=2), axis=0)  # (h*N, T_q, C/h)
            K_ = tf.concat(tf.split(K, n_heads, axis=2), axis=0)  # (h*N, T_k, C/h)
            V_ = tf.concat(tf.split(V, n_heads, axis=2), axis=0)  # (h*N, T_k, C/h)


            # Multiplication
            outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))  # (h*N, T_q, T_k)
            if relative_position:
                """only for self attention"""
                T_k, d_k = K_.get_shape().as_list()[1:]
                relative_position_enc_k = self._get_relative_position(n_steps=T_k, n_units=d_k, max_dist=max_dist, name='relative_position_encodings_key')  # [T_k, T_k, C/h]
                relative_position_enc_k = tf.matmul(tf.transpose(Q_, [1, 0, 2]), relative_position_enc_k, transpose_b=True)  # [T_q, h*N, T_k]
                relative_position_enc_k = tf.transpose(relative_position_enc_k, [1, 0, 2])  # [h*N, T_q, T_k]
                outputs += relative_position_enc_k

            # Scale
            outputs = outputs / (K_.get_shape().as_list()[-1] ** 0.5)  # (h*N, T_q, T_k)

            # Causality = Future blinding
            """only for self attention"""
            if causal:
                diag_vals = tf.ones_like(outputs[0, :, :])  # (T_q, T_k)
                tril_mask = tf.linalg.LinearOperatorLowerTriangular(diag_vals).to_dense()  # (T_q, T_k)
                tril_paddings = tf.ones_like(tril_mask) * (-2 ** 32 + 1)  # (T_q, T_k)
                tril_masking = lambda x: tf.where(tf.equal(tril_mask, 0), tril_paddings, x)
                outputs = tf.map_fn(tril_masking, outputs)  # (h*N, T_q, T_k)

            # mask out each query position from attending to itself
            if self_mask:
                diag = tf.zeros_like(outputs[:, :, 0])  # (T_q, T_k)
                outputs = tf.linalg.set_diag(input=outputs, diagonal=diag)

            # Activation
            outputs = tf.nn.softmax(outputs)  # (h*N, T_q, T_k)


            # Dropouts
            outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=is_training)  # (h*N, T_q, T_k)

            # Weighted sum
            if relative_position:
                """only for self attention"""
                T_v, d_v = V_.get_shape().as_list()[1:]
                relative_position_enc_v = self._get_relative_position(n_steps=T_v, n_units=d_v, max_dist=max_dist, name='relative_position_encodings_value')  # [T_v, T_v, C/h]
                relative_position_enc_v = tf.matmul(tf.transpose(outputs, [1, 0, 2]), relative_position_enc_v)  # [T_q, h*N, C/h]
                relative_position_enc_v = tf.transpose(relative_position_enc_v, [1, 0, 2])  # [h*N, T_q, C/h]
                outputs = tf.matmul(outputs, V_) + relative_position_enc_v  # ( h*N, T_q, C/h)

            else:
                outputs = tf.matmul(outputs, V_)  # ( h*N, T_q, C/h)

            # Restore shape
            outputs = tf.concat(tf.split(outputs, n_heads, axis=0), axis=2) # (N, T_q, C)

            # output projection
            outputs = tf.layers.dense(outputs, n_units)

            # Residual connection
            outputs += queries

            # Normalize
            outputs = self._normalize(outputs)  # (N, T_q, C)

        return outputs

    def _encode_segment_Time(self, inputs, n_units, dropout_rate, is_training):
        """inputs with shape = [batch_size, n_steps, feature_size]"""

        with tf.variable_scope("segment_encoding"):
            inputs_reshape = tf.reshape(inputs, shape=[-1, self._frequency_size, self._segment_width])  # [batch_size*n_steps, tonal_size, segment_width]
            inputs_reshape = tf.transpose(inputs_reshape, perm=[0, 2, 1])  # [batch_size*n_steps, segment_width, tonal_size]

            # Positional Encoding
            inputs_reshape += self._positional_encoding(batch_size=tf.shape(inputs_reshape)[0], time_steps=self._segment_width, n_units=self._frequency_size) * 0.01 + 0.01

            # Multihead attention
            inputs_reshape = self._multihead_attention(queries=inputs_reshape,
                                                       keys=inputs_reshape,
                                                       n_units=self._frequency_size,
                                                       n_heads=2,
                                                       activation_function=tf.nn.relu,
                                                       relative_position=True,
                                                       max_dist=4,
                                                       dropout_rate=dropout_rate,
                                                       is_training=is_training,
                                                       scope="frame_self_attention")
            ## Feed Forward
            inputs_reshape = self._feedforward(inputs_reshape, n_units=[self._frequency_size * 4, self._frequency_size], dropout_rate=dropout_rate, is_training=is_training)  # [batch_size*n_steps, segment_width, tonal_size]

            # restore shape
            inputs_reshape = tf.transpose(inputs_reshape, perm=[0, 2, 1])  # [batch_size*n_steps, tonal_size, segment_width]
            inputs_reshape = tf.reshape(inputs_reshape, shape=[-1, self._n_steps, self._frequency_size * self._segment_width])  # [batch_size, n_steps,  feature_size]

            # dense to input_embedding_size
            inputs_reshape = tf.layers.dropout(inputs_reshape, rate=dropout_rate, training=is_training)  # dropout
            segment_encodings = tf.layers.dense(inputs_reshape, n_units, activation=tf.nn.relu)  # [batch_size, n_steps, input_embedding_size]
            segment_encodings = self._normalize(segment_encodings)

        return segment_encodings

    def _encode_segment_Frequency(self, inputs, n_units, dropout_rate, is_training):
        """inputs with shape = [batch_size, n_steps, feature_size]"""

        with tf.variable_scope("segment_encoding"):
            inputs_reshape = tf.reshape(inputs, shape=[-1, self._frequency_size, self._segment_width])  # [batch_size*n_steps, tonal_size, segment_width]

            # Positional Encoding
            inputs_reshape += self._positional_encoding(batch_size=tf.shape(inputs_reshape)[0], time_steps=self._frequency_size, n_units=self._segment_width) * 0.01 + 0.01

            # Multihead attention
            inputs_reshape = self._multihead_attention(queries=inputs_reshape,
                                                       keys=inputs_reshape,
                                                       n_units=self._segment_width,
                                                       n_heads=1,
                                                       activation_function=tf.nn.relu,
                                                       relative_position=False,
                                                       max_dist=4,
                                                       dropout_rate=dropout_rate,
                                                       is_training=is_training,
                                                       scope="frame_self_attention")
            ## Feed Forward
            inputs_reshape = self._feedforward(inputs_reshape, n_units=[self._segment_width * 4, self._segment_width], dropout_rate=dropout_rate, is_training=is_training) # [batch_size*n_steps, tonal_size, segment_width]


            # restore shape
            inputs_reshape = tf.reshape(inputs_reshape, shape=[-1, self._n_steps, self._frequency_size * self._segment_width]) # [batch_size, n_steps, tonal_size*segment_width]

            # dense to input_embedding_size
            inputs_reshape = tf.layers.dropout(inputs_reshape, rate=dropout_rate, training=is_training)  # dropout
            segment_encodings = tf.layers.dense(inputs_reshape, n_units, activation=tf.nn.relu)  # [batch_size, n_steps, input_embedding_size]
            segment_encodings = self._normalize(segment_encodings)

        return segment_encodings

    def _binaryRound(self, x, cast_to_int=False):
        g = tf.get_default_graph()
        with ops.name_scope("BinaryRound") as name:
            if cast_to_int:
                with g.gradient_override_map({"Round": "Identity", "Cast": "Identity"}):
                    return tf.cast(tf.round(x), tf.int32, name=name)
            else:
                with g.gradient_override_map({"Round": "Identity"}):
                    return tf.round(x, name=name)

    def _chord_block_compression(self, hidden_states, chord_changes):
        """compress hidden states according to chord changes"""
        block_ids = tf.cumsum(chord_changes, axis=1)
        modify_ids = lambda x: tf.cond(tf.equal(x[0], 0), lambda: x, lambda: x - 1)
        block_ids = tf.map_fn(modify_ids, block_ids)
        block_ids.set_shape([None, self._n_steps])

        num_blocks = tf.reduce_max(block_ids, axis=1) + 1  # number of blocks of batched sequences
        max_steps = tf.reduce_max(num_blocks)  # max number of blocks

        segment_mean_and_pad = lambda x: tf.pad(tf.segment_mean(data=x[0], segment_ids=x[1]), paddings=[[0, max_steps - x[2]], [0, 0]], constant_values=0.0)
        chord_blocks = tf.map_fn(segment_mean_and_pad, (hidden_states, block_ids, num_blocks), dtype=tf.float32)

        return chord_blocks, block_ids, num_blocks

    def _decode_compressed_sequences(self, compressed_sequences, block_ids):
        # decode chord sequences according to chords_pred and block_ids
        gather_chords = lambda x: tf.gather(params=x[0], indices=x[1])
        chords_decode = tf.map_fn(gather_chords, (compressed_sequences, block_ids), dtype=compressed_sequences.dtype)

        return chords_decode

    def encoder(self, inputs, slope, dropout_rate, is_training):
        """inputs with shape = [batch_size, n_steps, feature_size]"""

        # Segment encoding
        with tf.variable_scope("encoder_segment_encodings"):
            segment_encodings_enc = self._encode_segment_Time(inputs, self._encoder_input_embedding_size, dropout_rate, is_training)  # [batch_size, n_steps, encoder_input_embedding_size]

        # Encoding
        with tf.variable_scope("encoder"):
            ''' encoder_inputs with shape = [batch_size, max_steps, n_inputs]'''
            encoder_inputs_embedded = segment_encodings_enc # compressed region, [batch_size, n_steps, encoder_input_embedding_size]

            ## Positional Encoding
            encoder_inputs_embedded += self._positional_encoding(batch_size=tf.shape(encoder_inputs_embedded)[0], time_steps=self._n_steps, n_units=self._encoder_input_embedding_size)

            ## Dropout
            encoder_inputs_embedded = tf.layers.dropout(encoder_inputs_embedded, rate=dropout_rate, training=is_training)

            ## Blocks
            s_task_enc = tf.nn.softmax(tf.get_variable('weights_of_layers_enc', dtype=tf.float32, shape=[2], initializer=tf.initializers.zeros()))  # [n_layers]
            weighted_hiddens_enc = tf.zeros(shape=tf.shape(encoder_inputs_embedded))
            for i in range(2):
                with tf.variable_scope("num_blocks_{}".format(i)):
                    ### Multihead Attention (self-attention)
                    encoder_inputs_embedded = self._multihead_attention(queries=encoder_inputs_embedded,
                                                                        keys=encoder_inputs_embedded,
                                                                        n_units=self._encoder_input_embedding_size,
                                                                        n_heads=8,
                                                                        relative_position=True,
                                                                        max_dist=16,
                                                                        dropout_rate=dropout_rate,
                                                                        is_training=is_training,
                                                                        scope="enc_self_attention")

                    ### Feed Forward, output shape = [batch_size, n_steps, encoder_input_embedding_size]
                    encoder_inputs_embedded = self._feedforward(encoder_inputs_embedded, n_units=[self._encoder_input_embedding_size * 4, self._encoder_input_embedding_size], dropout_rate=dropout_rate, is_training=is_training)

                    # weighted sum of all layers
                    weighted_hiddens_enc += (s_task_enc[i] * encoder_inputs_embedded)

            encoder_inputs_embedded = weighted_hiddens_enc

        chord_change_logits = tf.squeeze(tf.layers.dense(encoder_inputs_embedded, 1, activation=None)) # shape = [batch_size, n_steps]
        chord_change_prob = tf.sigmoid(slope * chord_change_logits)  # shape = [batch_size, n_steps]

        # Binarization
        chord_change_predictions = self._binaryRound(chord_change_prob, cast_to_int=True)  # Deterministic, shape = [batch_size, n_steps]

        return encoder_inputs_embedded, chord_change_logits, chord_change_predictions

    def decoder(self, inputs, encoder_inputs_embedded, chord_change_predictions, dropout_rate, is_training):
        # Segment encoding for Decoder
        with tf.variable_scope("decoder_segment_encodings"):
            segment_encodings_dec = self._encode_segment_Frequency(inputs, self._decoder_input_embedding_size, dropout_rate, is_training) # [batch_size, n_steps, decoder_input_embedding_size]

            segment_encodings_dec_blocked, block_ids, num_blocks = self._chord_block_compression(segment_encodings_dec, chord_change_predictions)
            segment_encodings_dec_blocked = self._decode_compressed_sequences(segment_encodings_dec_blocked, block_ids)  # shape = [batch_size, n_steps, decoder_input_embedding_size]
            segment_encodings_dec_blocked.set_shape([None, self._n_steps, self._decoder_input_embedding_size])

        # Decoding
        with tf.variable_scope("decoder"):
            decoder_inputs_embedded = segment_encodings_dec + segment_encodings_dec_blocked + encoder_inputs_embedded  # [batch_size, n_steps, 3*decoder_input_embedding_size]

            ## Positional Encoding
            decoder_inputs_embedded += self._positional_encoding(batch_size=tf.shape(decoder_inputs_embedded)[0], time_steps=self._n_steps, n_units=self._decoder_input_embedding_size)

            ## Dropout
            decoder_inputs_embedded = tf.layers.dropout(decoder_inputs_embedded, rate=dropout_rate, training=is_training)

            ## Blocks
            s_task_dec = tf.nn.softmax(tf.get_variable('weights_of_layers_dec', dtype=tf.float32, shape=[2], initializer=tf.initializers.zeros()))  # [n_layers]
            weighted_hiddens_dec = tf.zeros(shape=tf.shape(decoder_inputs_embedded))
            for i in range(2):
                with tf.variable_scope("num_blocks_{}".format(i)):
                    ### Multihead Attention (self-attention)
                    decoder_inputs_embedded = self._multihead_attention(queries=decoder_inputs_embedded,
                                                                        keys=decoder_inputs_embedded,
                                                                        n_units=self._decoder_input_embedding_size,
                                                                        n_heads=8,
                                                                        relative_position=True,
                                                                        max_dist=16,
                                                                        self_mask=False,
                                                                        dropout_rate=dropout_rate,
                                                                        is_training=is_training,
                                                                        scope="dec_self_attention")

                    ### Multihead Attention (seq2seq attention)
                    decoder_inputs_embedded = self._multihead_attention(queries=decoder_inputs_embedded,
                                                                        keys=encoder_inputs_embedded,
                                                                        n_units=self._decoder_input_embedding_size,
                                                                        n_heads=8,
                                                                        relative_position=False,
                                                                        max_dist=16,
                                                                        dropout_rate=dropout_rate,
                                                                        is_training=is_training,
                                                                        scope="encoder_decoder_attention")

                    ### Feed Forward, output shape = [batch_size, n_steps, decoder_input_embedding_size]
                    decoder_inputs_embedded = self._feedforward(decoder_inputs_embedded, n_units=[self._decoder_input_embedding_size * 4, self._decoder_input_embedding_size], dropout_rate=dropout_rate, is_training=is_training)

                    # weighted sum of all layers
                    weighted_hiddens_dec += (s_task_dec[i] * decoder_inputs_embedded)

            decoder_inputs_embedded = weighted_hiddens_dec

        # Outputt preojection
        logits = tf.layers.dense(decoder_inputs_embedded, self._n_classes)  # shape = [batch_size, n_steps, n_classes]
        chord_predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)  # shape = [batch_size, n_steps]

        return logits, chord_predictions

    def load_data(self):
        fileDir = 'preprocessed_data\\Billboard_data_mirex_Mm_model_input_final.npz'
        with np.load(fileDir) as input_data:
            x_train = input_data['x_train']
            TC_train = input_data['TC_train']
            y_train = input_data['y_train']
            y_cc_train = input_data['y_cc_train']
            y_len_train = input_data['y_len_train']
            x_valid = input_data['x_valid']
            TC_valid = input_data['TC_valid']
            y_valid = input_data['y_valid']
            y_cc_valid = input_data['y_cc_valid']
            y_len_valid = input_data['y_len_valid']
            split_sets = input_data['split_sets']
        split_sets = split_sets.item()

        return x_train, TC_train, y_train, y_cc_train, y_len_train, \
               x_valid, TC_valid, y_valid, y_cc_valid, y_len_valid, \
               split_sets

    def train(self):
        # load input data
        print("load intpu data...")
        x_train, TC_train, y_train, y_cc_train, y_len_train, \
        x_valid, TC_valid, y_valid, y_cc_valid, y_len_valid, \
        split_sets = self.load_data()
        num_examples_train = x_train.shape[0]

        # Define placeholders
        print("build model...")
        x = tf.placeholder(tf.float32, [None, self._n_steps, self._feature_size], name='encoder_inputs') # shape = [batch_size, n_steps, n_inputs]
        y = tf.placeholder(tf.int32, [None, self._n_steps], name='chord_labels') # ground_truth, shape = [batch_size, n_steps]
        y_cc = tf.placeholder(tf.int32, [None, self._n_steps], name='chord_change_labels') # ground_truth, shape = [batch_size, n_steps]
        y_len = tf.placeholder(tf.int32, shape=[None], name="sequence_lengths")
        dropout_rate = tf.placeholder(tf.float32, name='dropout_rate')
        is_training = tf.placeholder(tf.bool, name='is_training')
        global_step = tf.placeholder(tf.int32, name='global_step')
        slope = tf.placeholder(tf.float32, name='slope')
        stochastic_tensor = tf.placeholder(tf.bool, name='stochastic_tensor')

        encoder_inputs_embedded, chord_change_logits, chord_change_predictions = self.encoder(x, slope, dropout_rate, is_training)
        logits, chord_predictions = self.decoder(x, encoder_inputs_embedded, chord_change_predictions, dropout_rate, is_training)

        # Define loss
        with tf.name_scope('loss'):
            loss_ct = self._lambda_loss_ct * tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.cast(y_cc, tf.float32), logits=chord_change_logits))
            loss_c = self._lambda_loss_c * tf.losses.softmax_cross_entropy(onehot_labels=tf.one_hot(y, depth=self._n_classes), logits=logits, label_smoothing=0.1)

            # L2 norm regularization
            vars = tf.trainable_variables()
            L2_regularizer = self._lambda_L2 * tf.add_n([tf.nn.l2_loss(v) for v in vars if 'bias' not in v.name])

            # loss
            loss = loss_ct + loss_c + L2_regularizer

        with tf.name_scope('optimization'):
            # apply learning rate decay
            learning_rate = tf.train.exponential_decay(learning_rate=self._initial_learning_rate,
                                                       global_step=global_step,
                                                       decay_steps=(x_train.shape[0] // self._batch_size),
                                                       decay_rate=0.96,
                                                       staircase=True)

            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                               beta1=0.9,
                                               beta2=0.98,
                                               epsilon=1e-9)

            # Apply gradient clipping
            gvs = optimizer.compute_gradients(loss)
            capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) if grad is not None else (grad, var) for grad, var in gvs]
            train_op = optimizer.apply_gradients(capped_gvs)

        # Define accuracy
        with tf.name_scope('accuracy'):
            label_mask = tf.less(y, 24)  # where label != 24('X)' and label != 25('pad')
            correct_predictions = tf.equal(chord_predictions, y)
            correct_predictions_mask = tf.boolean_mask(tensor=correct_predictions, mask=label_mask)
            accuracy = tf.reduce_mean(tf.cast(correct_predictions_mask, tf.float32))

        # Training
        print('train the model...')
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            epoch = num_examples_train // self._batch_size # steps per epoch
            annealing_slope = 1.0
            for step in range(self._training_steps):

                if step % epoch == 0:
                    # shuffle trianing set
                    indices = random.sample(range(num_examples_train), num_examples_train)
                    batch_indices = [indices[x:x + self._batch_size] for x in range(0, len(indices), self._batch_size)]

                    if step != 0:
                        annealing_slope *= self._annealing_rate

                # training
                batch = (x_train[batch_indices[step % len(batch_indices)]],
                         y_cc_train[batch_indices[step % len(batch_indices)]],
                         y_train[batch_indices[step % len(batch_indices)]],
                         y_len_train[batch_indices[step % len(batch_indices)]],
                         TC_train[batch_indices[step % len(batch_indices)]])

                x_batch = batch[0]
                train_run_list = [train_op, loss, loss_ct, loss_c, L2_regularizer, chord_change_predictions, chord_predictions, accuracy]
                train_feed_fict = {x: x_batch,
                                   y_cc: batch[1],
                                   y: batch[2],
                                   y_len: batch[3],
                                   dropout_rate: self._dropout_rate,
                                   is_training: True,
                                   global_step: step + 1,
                                   slope: annealing_slope,
                                   stochastic_tensor: True}
                _, train_loss, train_loss_ct, train_loss_c, train_L2,  train_cc_pred, train_c_pred, train_acc = sess.run(train_run_list, feed_dict=train_feed_fict)

                if step % (epoch // 2) == 0:
                    print("------ step %d: train_loss %.4f (ct %.4f, c %.4f, L2 %.4f), train_accuracy %.4f ------" % (step, train_loss, train_loss_ct, train_loss_c, train_L2, train_acc))

if __name__ == '__main__':

    model = Harmony_Transformer()
    model.train()



