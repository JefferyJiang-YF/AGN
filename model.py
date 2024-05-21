# -*- coding: utf-8 -*-

import os
import random
from typing import Dict, Optional, Tuple, List, Callable, Union

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from langml import keras, K, L
from langml.plm.bert import load_bert
from langml.utils import pad_sequences
from langml.layers import (
    CRF, LayerNorm,
    ConditionalLayerNormalization
)
from langml.tensor_typing import Tensors, Initializer, Activation


# set random seed
seed_value = int(os.getenv('RANDOM_SEED', -1))
if seed_value != -1:
    random.seed(seed_value)
    np.random.seed(seed_value)
    tf.random.set_seed(seed_value)


def sequence_masking(x: Tensors,
                     mask: Optional[Tensors] = None,
                     value: Union[str, float] = '-inf',
                     axis: Optional[int] = None) -> Tensors:
    """mask sequence
    Args:
        x: input tensor
        mask: mask of input tensor
    """
    if mask is None:
        return x
    if isinstance(value, str):
        assert value in ['-inf', 'inf'], 'if value is a str, please choose it from [`-inf`, `inf`]'
    x_dtype = K.dtype(x)
    if x_dtype == 'bool':
        x = K.cast(x, 'int32')
    if K.dtype(mask) != K.dtype(x):
        mask = K.cast(mask, K.dtype(x))
    if value == '-inf':
        value = -1e12
    elif value == 'inf':
        value = 1e12
    if axis is None:
        axis = 1
    elif axis < 0:
        axis = K.ndim(x) + axis
    assert axis > 0, 'axis must be greater than 0'
    mask = align(mask, [0, axis], K.ndim(x))
    value = K.cast(value, K.dtype(x))
    x = x * mask + value * (1 - mask)
    if x_dtype == 'bool':
        x = K.cast(x, 'bool')
    return x


class WithVAELoss(L.Layer):
    def call(self, inputs):
        z_mean, z_log_var, y_true, y_pred = inputs
        reconstruction_loss = K.mean(
            keras.losses.binary_crossentropy(y_true, y_pred)
        )
        kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
        kl_loss = K.mean(kl_loss)
        kl_loss *= -0.5
        loss =  reconstruction_loss + kl_loss
        self.add_loss(loss, inputs=inputs)
        return y_pred

    def compute_mask(self, inputs, mask=None):
        return mask

    def compute_output_shape(self, input_shape):
        return input_shape[-1]

    @staticmethod
    def get_custom_objects():
        return {'WithMLMSparseCategoricalCrossEntropy': WithMLMSparseCategoricalCrossEntropy}


class Sampling(L.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = K.int_shape(z_mean)[0]
        dim = K.int_shape(z_mean)[1]
        epsilon = K.random_normal(shape=(batch, dim))
        return z_mean + K.exp(0.5 * z_log_var) * epsilon
    
    def compute_output_shape(self, input_shape):
        return input_shape[0]


class VariationalAutoencoder:
    def __init__(self, latent_dim=64, hidden_dim=128, activation='relu', epochs=10, batch_size=64):
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.activation = activation
        self.epochs = epochs
        self.batch_size = batch_size
        self.autoencoder = None
        self.encoder = None
        self.decoder = None
        self.his = None

    def _compile(self, input_dim):
        """
        compile the computational graph
        """
        input_vec = L.Input(batch_shape=(self.batch_size, input_dim), name='input', dtype='float32')
        hidden = L.Dense(self.hidden_dim, activation=self.activation, name='hidden')(input_vec)
        z_mean = L.Dense(self.latent_dim, name='z_mean')(hidden)
        z_log_var = L.Dense(self.latent_dim, name='z_log_var')(hidden)
        encoded = Sampling(name='sampling')([z_mean, z_log_var])
        decoded = L.Dense(input_dim, activation="sigmoid", name='decoder')(encoded)
        # with vae loss
        decoded = WithVAELoss()([z_mean, z_log_var, input_vec, decoded])

        self.autoencoder = keras.Model(input_vec, decoded)
        self.encoder = keras.Model(input_vec, encoded)
        self.autoencoder.compile(optimizer='adam', loss=None) 
        self.autoencoder.summary()

    def fit(self, X, verbose=2):
        if not self.autoencoder:
            self._compile(X.shape[1])
        per_size = int(X.shape[0] * 0.9) // self.batch_size
        train_size = int((per_size + 1) * self.batch_size)
        X_shuffle = shuffle(X)
        X_train = X_shuffle[:train_size]
        X_test = X_shuffle[train_size:]
        print(">> train shape:", X_train.shape)
        print(">> dev shape:", X_test.shape)
        # X_train = pad_sequences(X_train)
        # X_test = pad_sequences(X_test)
        self.autoencoder.fit(X_train, X_train,
                             epochs=self.epochs,
                             batch_size=self.batch_size,
                             shuffle=True,
                             callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)],
                             validation_data=(X_test, X_test), verbose=verbose
                            )


class Autoencoder:
    def __init__(self, latent_dim=64, activation='relu', epochs=10, batch_size=64):
        self.latent_dim = latent_dim
        self.activation = activation
        self.epochs = epochs
        self.batch_size = batch_size
        self.autoencoder = None
        self.encoder = None
        self.decoder = None
        self.his = None

    def _compile(self, input_dim):
        """
        compile the computational graph
        """
        input_vec = L.Input(shape=(input_dim,))
        encoded = L.Dense(self.latent_dim, activation=self.activation)(input_vec)
        decoded = L.Dense(input_dim, activation=self.activation)(encoded)
        self.autoencoder = keras.Model(input_vec, decoded)
        self.encoder = keras.Model(input_vec, encoded)
        self.autoencoder.compile(optimizer='adam', loss='mean_squared_error')

    def fit(self, X, verbose=2):
        if not self.autoencoder:
            self._compile(X.shape[1])
        X_train, X_test = train_test_split(X)
        self.autoencoder.fit(X_train, X_train,
                             epochs=self.epochs,
                             batch_size=self.batch_size,
                             shuffle=True,
                             callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)],
                             validation_data=(X_test, X_test), verbose=verbose)


class GatedLinearUnit(L.Layer):
    def __init__(self,
                 units: int,
                 kernel_initializer: Initializer = 'glorot_normal',
                 **kwargs):
        super(GatedLinearUnit, self).__init__(**kwargs)
        self.units = units
        self.kernel_initializer = kernel_initializer
        self.supports_masking = True

    def get_config(self) -> dict:
        config = {
            "units": self.units,
            "kernel_initializer": keras.initializers.serialize(self.kernel_initializer),
        }
        base_config = super(GatedLinearUnit, self).get_config()

        return dict(base_config, **config)

    def build(self, input_shape: Tensors):
        super(GatedLinearUnit, self).build(input_shape)
        self.linear = L.Dense(self.units, kernel_initializer=self.kernel_initializer, name='dense-t')
        self.sigmoid = L.Dense(self.units, activation='sigmoid',
                               kernel_initializer=self.kernel_initializer, name='dense-g')

    def call(self, inputs: Tensors, mask: Optional[Tensors] = None):
        return self.linear(inputs) * self.sigmoid(inputs)

    def compute_output_shape(self, input_shape):
        return input_shape

    def compute_mask(self, inputs: Tensors, mask: Optional[Tensors] = None):
        if isinstance(mask, list):
            mask = mask[0]
        return mask

    @staticmethod
    def get_custom_objects() -> dict:
        return {'GatedLinearUnit': GatedLinearUnit}


class SelfAttention(L.Layer):
    def __init__(self,
                 units: Optional[int] = None,
                 return_attention: bool = False,
                 is_residual: bool = True,
                 activation: Optional[Activation] = None,
                 kernel_initializer: Initializer = 'glorot_normal',
                 use_bias: bool = True,
                 dropout_rate: float = 0.0,
                 **kwargs):
        super(SelfAttention, self).__init__(**kwargs)

        self.supports_masking = True

        self.units = units
        self.return_attention = return_attention
        self.is_residual = is_residual
        self.activation = keras.activations.get(activation)
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.use_bias = use_bias
        self.dropout_rate = dropout_rate

    def get_config(self) -> dict:
        config = {
            "units": self.units,
            "return_attention": self.return_attention,
            "is_residual": self.is_residual,
            "activation": keras.activations.serialize(self.activation),
            "kernel_initializer": keras.initializers.serialize(self.kernel_initializer),
            "use_bias": self.use_bias,
            "dropout_rate": self.dropout_rate
        }
        base_config = super(SelfAttention, self).get_config()

        return dict(base_config, **config)

    def build(self, input_shape: Tensors):
        super(SelfAttention, self).build(input_shape)

        if isinstance(input_shape, list):
            feature_dim = int(input_shape[-1][-1])
        else:
            feature_dim = int(input_shape[-1])

        units = feature_dim if self.units is None else self.units

        self.q_dense = L.Dense(
            units=units,
            activation=self.activation,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            name='dense-q',
        )
        self.k_dense = L.Dense(
            units=units,
            activation=self.activation,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            name='dense-k',
        )
        self.v_dense = L.Dense(
            units=units,
            activation=self.activation,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            name='dense-v',
        )
        self.o_dense = L.Dense(
            units=feature_dim,
            activation=self.activation,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            name='dense-o',
        )
        if self.is_residual:
            self.glu = GatedLinearUnit(
                feature_dim, kernel_initializer=self.kernel_initializer, name='glu')
            self.layernorm = LayerNorm(name='layernorm')

    def call(self, inputs: Tensors, mask: Optional[Tensors] = None, **kwargs) -> Union[List[Tensors], Tensors]:
        if isinstance(inputs, list):
            q, k, v = inputs
        else:
            q = k = v = inputs

        mask = mask[0] if isinstance(mask, list) else mask

        qw = self.q_dense(q)
        kw = self.k_dense(k)
        vw = self.v_dense(v)

        e = K.batch_dot(qw, kw, axes=2)
        e /= K.int_shape(qw)[-1]**0.5
        # axis=1 if channel_last else 2
        e = sequence_masking(e, mask, '-inf', 1)
        a = K.softmax(e)
        if self.dropout_rate:
            a = L.Dropout(self.dropout_rate)(a)
        out = K.batch_dot(a, vw)
        out = self.o_dense(out)
        if self.is_residual:
            # out += qw
            if self.dropout_rate > 0:
                out = L.Dropout(self.dropout_rate)(out)
            out = out + self.glu(qw)
            out = self.layernorm(out)
        if self.return_attention:
            return [out, a]
        return out

    def compute_mask(self,
                     inputs: Tensors,
                     mask: Optional[Tensors] = None) -> Union[List[Union[Tensors, None]], Tensors]:
        mask = mask[0] if isinstance(mask, list) else mask
        if self.return_attention:
            return [mask, None]
        return mask

    def compute_output_shape(self, input_shape: Tensors) -> Union[List[Tensors], Tensors]:
        if not isinstance(input_shape, list):
            output_shape = input_shape[0]
        else:
            output_shape = input_shape
        if self.return_attention:
            attention_shape = (output_shape[0], output_shape[1], output_shape[1])
            return [output_shape, attention_shape]
        return output_shape

    @staticmethod
    def get_custom_objects() -> dict:
        return {'SelfAttention': SelfAttention}


class AGN(L.Layer):
    def __init__(self,
                 activation='swish',
                 attn_initializer=None,
                 dropout_rate=0.1,
                 valve_rate=0.3,
                 dynamic_valve=False,
                 **kwargs):
        super(AGN, self).__init__(**kwargs)
        self.activation = activation
        self.attn_initializer = attn_initializer
        self.dropout_rate = dropout_rate
        self.valve_rate = valve_rate
        self.dynamic_valve = dynamic_valve
        self.supports_masking = True

    def build(self, input_shape):
        feature_size = input_shape[0][-1]
        self.valve_transform = L.Dense(feature_size, activation='sigmoid', use_bias=False, name='valve')
        self.dynamic_valve = L.Dropout(1.0 - self.valve_rate)
        self.dropout = L.Dropout(self.dropout_rate)
        self.attn = SelfAttention(activation=self.activation,
                                  kernel_initializer=self.attn_initializer,
                                  is_residual=True,
                                  return_attention=True,
                                  dropout_rate=self.dropout_rate,
                                  name='attn')
        super(AGN, self).build(input_shape)

    def call(self, inputs):
        X, gi = inputs
        valve = self.valve_transform(X)
        if self.dynamic_valve:
            valve = self.dynamic_valve(valve)
        else:
            valve = L.Lambda(lambda x: x * tf.where(
                tf.math.logical_and(x > 0.5 - self.valve_rate, x < 0.5 + self.valve_rate), x=1.0, y=0.0))(valve)
        enhanced = X + valve * gi
        enhanced = self.dropout(enhanced)
        return self.attn(enhanced)

    def compute_output_shape(self, input_shape):
        return [(input_shape[0][0], input_shape[0][1], input_shape[0][2]),
                (input_shape[0][0], input_shape[0][1], input_shape[0][1])]

    def compute_mask(self, inputs, mask=None):
        return [mask, None]


class AGNModel:
    """ Adaptive Gate Network
    """
    def __init__(self, config, task='clf'):
        self.config = config
        # load pretrained bert
        self.model = None
        self.attn_model = None
        self.build(task=task)

    def build(self, task='clf'):
        assert task in ['clf', 'ner', 'sts'], 'please specify task from [`clf`, `ner`, `sts`]'

        bert_model, _ = load_bert(
            config_path=os.path.join(self.config['pretrained_model_dir'], 'bert_config.json'),
            checkpoint_path=os.path.join(self.config['pretrained_model_dir'], 'bert_model.ckpt'),
        )
        # GI
        gi_in = L.Input(name="Input-GI", shape=(self.config["ae_latent_dim"], ), dtype="float32")
        gi = gi_in

        # AGN
        X = bert_model.output
        feature_size = K.int_shape(X)[-1]
        C = L.Lambda(lambda x: x[:, 0], name='C')(X)

        gi = L.Dense(feature_size)(gi)  # (B, D)
        gi = L.Dropout(self.config.get('dropout_rate', 0.1))(gi)
        gi = L.Lambda(lambda x: K.expand_dims(x, 1))(gi)  # (B, 1, D)

        X, attn_weight = AGN(
            name='AGN',
            activation='swish',
            dropout_rate=self.config.get('dropout_rate', 0.1),
            valve_rate=self.config.get('valve_rate', 0.3),
            dynamic_valve=self.config.get('use_dynamic_valve', False),
        )([X, gi])

        self.attn_model = keras.Model(inputs=(*bert_model.input, gi_in), outputs=attn_weight)

        if task == 'clf':
            # fuse
            maxpool = L.GlobalMaxPooling1D()(X)
            maxpool = L.Dense(feature_size)(maxpool)
            # maxpool = L.Dropout(self.config.get('dropout_rate', 0.1))(maxpool)
            C = L.Dense(feature_size)(C)
            # C = L.Dropout(self.config.get('dropout_rate', 0.1))(C)
            output = ConditionalLayerNormalization()([C, maxpool])

            # output
            output = L.Dense(self.config.get('hidden_size', 256), activation='swish')(output)
            output = L.Dropout(self.config.get('dropout_rate', 0.1))(output)
            output = L.Dense(self.config['output_size'], activation='softmax')(output)
            self.model = keras.Model(inputs=(*bert_model.input, gi_in), outputs=output)
        elif task == 'ner':
            crf = CRF(self.config['output_size'], sparse_target=False, name='crf')
            if self.config.get('use_agn', True):
                print('>>> apply agn...')
                output = L.Dropout(self.config.get('dropout_rate', 0.1))(X)
            else:
                print('>>> apply bert...')
                output = L.Dropout(self.config.get('dropout_rate', 0.1))(bert_model.output)
            output = L.Dense(self.config['output_size'], name='tag')(output)
            output = crf(output)
            # output = L.Dense(self.config['output_size'], activation='softmax', name='tag')(output)
            self.model = keras.Model(inputs=(*bert_model.input, gi_in), outputs=output)
        elif task == 'sts':
            pass

        if self.config['optimizer'] == 'adamw':
            lr_schedule = tf.optimizers.schedules.ExponentialDecay(self.config['learning_rate'], 100, 0.9)
            wd_schedule = tf.optimizers.schedules.ExponentialDecay(5e-5, 100, 0.9)
            optimizer = tfa.optimizers.AdamW(learning_rate=lr_schedule, weight_decay=lambda : None)
            optimizer.weight_decay = lambda : wd_schedule(optimizer.iterations)
        else:
            optimizer = keras.optimizers.Adam(self.config['learning_rate'])
        self.model.compile(
            loss='sparse_categorical_crossentropy' if task == 'clf' else crf.loss,
            optimizer=optimizer,
        )
        # self.model.summary()

    def fit(self, *args, **kwargs):
        return self.model.fit(*args, **kwargs)
