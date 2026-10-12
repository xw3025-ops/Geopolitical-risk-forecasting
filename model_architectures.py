"""
Shared model-building functions for the two release v1.0 models, factored
out so the inference pipeline doesn't duplicate the architecture code from
release_v1.0_goldstein.py / release_v1.0_direction.py. Weights are loaded
with model.load_weights(...) after building (not keras.models.load_model),
since these models use custom Lambda reshaping that Keras can't always
reconstruct automatically from a saved config - rebuilding the architecture
fresh in code and loading just the numbers is the reliable path.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras

D_MODEL = 64
NUM_HEADS = 4
FF_DIM = 128
COUNTRY_EMBED_DIM = 8
NUM_TEMPORAL_BLOCKS = 2
NUM_CROSSDYAD_BLOCKS = 2

RELATION_NAMES = ["self", "reverse", "same_source", "same_target",
                   "forward_chain", "backward_chain", "unrelated"]
N_RELATIONS = len(RELATION_NAMES)


def swiglu_ffn(x, name, dropout=0.1):
    gate = keras.layers.Dense(FF_DIM, name=f"{name}_gate")(x)
    gate = keras.layers.Activation("swish", name=f"{name}_swish")(gate)
    value = keras.layers.Dense(FF_DIM, name=f"{name}_value")(x)
    hidden = keras.layers.Multiply(name=f"{name}_gated")([gate, value])
    out = keras.layers.Dense(D_MODEL, name=f"{name}_out")(hidden)
    out = keras.layers.Dropout(dropout)(out)
    return out


class StructuralBiasAttention(keras.layers.Layer):
    def __init__(self, num_heads, key_dim, relation_matrix, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.dropout_rate = dropout
        self.relation_matrix = tf.constant(relation_matrix, dtype=tf.int32)

    def build(self, input_shape):
        d_model = input_shape[-1]
        proj_dim = self.num_heads * self.key_dim
        init = keras.initializers.GlorotUniform()
        self.wq = self.add_weight(shape=(d_model, proj_dim), initializer=init, name="wq")
        self.wk = self.add_weight(shape=(d_model, proj_dim), initializer=init, name="wk")
        self.wv = self.add_weight(shape=(d_model, proj_dim), initializer=init, name="wv")
        self.wo = self.add_weight(shape=(proj_dim, d_model), initializer=init, name="wo")
        self.relation_bias = self.add_weight(
            shape=(N_RELATIONS, self.num_heads), initializer="zeros", name="relation_bias"
        )
        self.dropout = keras.layers.Dropout(self.dropout_rate)
        self.built = True

    def call(self, x, training=None):
        batch = tf.shape(x)[0]
        n = tf.shape(x)[1]

        def split_heads(t):
            t = tf.reshape(t, [batch, n, self.num_heads, self.key_dim])
            return tf.transpose(t, [0, 2, 1, 3])

        q = split_heads(tf.matmul(x, self.wq))
        k = split_heads(tf.matmul(x, self.wk))
        v = split_heads(tf.matmul(x, self.wv))
        scores = tf.matmul(q, k, transpose_b=True) / tf.sqrt(tf.cast(self.key_dim, tf.float32))
        bias = tf.gather(self.relation_bias, self.relation_matrix)
        bias = tf.transpose(bias, [2, 0, 1])
        scores = scores + bias[tf.newaxis, :, :, :]
        weights = tf.nn.softmax(scores, axis=-1)
        weights = self.dropout(weights, training=training)
        out = tf.matmul(weights, v)
        out = tf.transpose(out, [0, 2, 1, 3])
        out = tf.reshape(out, [batch, n, self.num_heads * self.key_dim])
        return tf.matmul(out, self.wo)


def _temporal_block_plain(x, name, dropout=0.1):
    attn_out = keras.layers.MultiHeadAttention(
        num_heads=NUM_HEADS, key_dim=D_MODEL // NUM_HEADS, dropout=dropout, name=f"{name}_mha"
    )(x, x)
    x = keras.layers.LayerNormalization(name=f"{name}_ln1")(x + attn_out)
    ffn_out = swiglu_ffn(x, name=f"{name}_ffn", dropout=dropout)
    x = keras.layers.LayerNormalization(name=f"{name}_ln2")(x + ffn_out)
    return x


def _crossdyad_block_structural(x, relation_matrix, name):
    attn_out = StructuralBiasAttention(
        num_heads=NUM_HEADS, key_dim=D_MODEL // NUM_HEADS,
        relation_matrix=relation_matrix, dropout=0.1, name=f"{name}_structattn"
    )(x)
    x = keras.layers.LayerNormalization(name=f"{name}_ln1")(x + attn_out)
    ffn_out = swiglu_ffn(x, name=f"{name}_ffn")
    x = keras.layers.LayerNormalization(name=f"{name}_ln2")(x + ffn_out)
    return x


def _shared_backbone(numeric_seq_input, actor1_input, actor2_input,
                      n_dyads, lookback_weeks, n_countries, dropout=0.1):
    token = keras.layers.Dense(D_MODEL, activation="relu", name="token_projection")(numeric_seq_input)

    positions = keras.layers.Lambda(
        lambda t: tf.range(lookback_weeks), output_shape=(lookback_weeks,)
    )(numeric_seq_input)
    temporal_pos_embedding = keras.layers.Embedding(
        input_dim=lookback_weeks, output_dim=D_MODEL, name="temporal_position_embedding"
    )(positions)
    token = keras.layers.Lambda(lambda t: t[0] + t[1], name="add_temporal_position")(
        [token, temporal_pos_embedding]
    )

    merged = keras.layers.Lambda(
        lambda t: tf.reshape(t, [-1, lookback_weeks, D_MODEL]), name="merge_batch_dyad"
    )(token)

    x = merged
    for i in range(NUM_TEMPORAL_BLOCKS):
        x = _temporal_block_plain(x, name=f"temporal_block{i}", dropout=dropout)

    temporal_summary = keras.layers.Lambda(lambda t: t[:, -1, :], name="take_last_week")(x)
    temporal_summary = keras.layers.Lambda(
        lambda t: tf.reshape(t, [-1, n_dyads, D_MODEL]), name="split_batch_dyad"
    )(temporal_summary)

    country_embedding = keras.layers.Embedding(
        input_dim=n_countries, output_dim=COUNTRY_EMBED_DIM, name="country_embedding"
    )
    actor1_vec = country_embedding(actor1_input)
    actor2_vec = country_embedding(actor2_input)

    dyad_token = keras.layers.Concatenate(axis=-1)([temporal_summary, actor1_vec, actor2_vec])
    dyad_token = keras.layers.Dense(D_MODEL, activation="relu", name="dyad_token_projection")(dyad_token)
    return dyad_token


def build_goldstein_model(n_dyads, lookback_weeks, n_numeric, n_countries,
                          dropout=0.1, residual_baseline=False):
    """Goldstein regression: plain cross-dyad attention, linear output.
    The defaults rebuild release v1.0 exactly. residual_baseline=True adds a
    fourth input, "baseline" (batch, n_dyads) - a forecast known at
    prediction time, e.g. the trailing 26-week mean - and the network
    predicts only the correction to it (output = baseline + network)."""
    numeric_seq_input = keras.layers.Input(shape=(n_dyads, lookback_weeks, n_numeric), name="numeric_seq")
    actor1_input = keras.layers.Input(shape=(n_dyads,), dtype="int32", name="actor1_idx")
    actor2_input = keras.layers.Input(shape=(n_dyads,), dtype="int32", name="actor2_idx")
    inputs = [numeric_seq_input, actor1_input, actor2_input]

    dyad_token = _shared_backbone(numeric_seq_input, actor1_input, actor2_input,
                                   n_dyads, lookback_weeks, n_countries, dropout=dropout)

    y = dyad_token
    for i in range(NUM_CROSSDYAD_BLOCKS):
        y = _temporal_block_plain(y, name=f"crossdyad_block{i}", dropout=dropout)

    output = keras.layers.Dense(1, name="per_dyad_output")(y)
    output = keras.layers.Reshape((n_dyads,), name="squeeze")(output)

    if residual_baseline:
        baseline_input = keras.layers.Input(shape=(n_dyads,), name="baseline")
        inputs.append(baseline_input)
        output = keras.layers.Add(name="add_baseline")([baseline_input, output])

    return keras.Model(inputs=inputs, outputs=output)


def build_direction_model(n_dyads, lookback_weeks, n_numeric, n_countries, relation_matrix):
    """release_v1.0_direction: structural-bias cross-dyad attention, 2-way softmax output."""
    numeric_seq_input = keras.layers.Input(shape=(n_dyads, lookback_weeks, n_numeric), name="numeric_seq")
    actor1_input = keras.layers.Input(shape=(n_dyads,), dtype="int32", name="actor1_idx")
    actor2_input = keras.layers.Input(shape=(n_dyads,), dtype="int32", name="actor2_idx")

    dyad_token = _shared_backbone(numeric_seq_input, actor1_input, actor2_input,
                                   n_dyads, lookback_weeks, n_countries)

    y = dyad_token
    for i in range(NUM_CROSSDYAD_BLOCKS):
        y = _crossdyad_block_structural(y, relation_matrix, name=f"crossdyad_block{i}")

    output = keras.layers.Dense(2, activation="softmax", name="per_dyad_output")(y)

    return keras.Model(inputs=[numeric_seq_input, actor1_input, actor2_input], outputs=output)
