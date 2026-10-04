"""
Train the spatio-temporal transformer on the 21-country panel (20 MENA + USA,
441 dyads), using the FULL BigQuery feature set (CAMEO counts, category
rollups, goldstein stats, news tone/sentiment, media coverage volume, actor
diversity) instead of the smaller count_01..20 + avg_goldstein set - and add
a learned STRUCTURAL BIAS to the cross-dyad attention stage.

Structural bias: plain self-attention treats every other dyad as equally
unrelated a priori. But e.g. dyad AB and dyad AC obviously aren't "equally
unrelated" to each other the way AB and some totally disjoint dyad are - AB
and AC share country A. We tell the model WHICH structural relationship
holds between any two dyads (it doesn't have to infer this from content
alone), and let it learn HOW MUCH each relationship type should matter, by
adding a per-relation-type learned scalar to the raw attention score before
softmax (one bias value per attention head, per block):

    score_ij = (Q_i . K_j) / sqrt(d)  +  bias[relation_type(i, j)]

Relation types (checked in this precedence order; a self-dyad like USA<->USA
can satisfy more than one, so order matters):
  self           - i and j are the same dyad
  reverse        - j is the reverse of i (AB vs BA)
  same_source    - i and j share Actor1 (AB vs AC)
  same_target    - i and j share Actor2 (AB vs CB)
  forward_chain  - i's target is j's source (AB vs BC: A->B->C)
  backward_chain - j's target is i's source (AB vs CA: C->A->B)
  unrelated      - no shared country at all (the large majority of pairs)

This can't be expressed with Keras's built-in MultiHeadAttention (it only
supports boolean masking, not an arbitrary learned additive bias), so the
cross-dyad stage uses a small custom attention layer below. The temporal
stage (self-attention across a dyad's own history) is unchanged - structural
bias is specifically about dyad-to-dyad relationships, which doesn't apply
there.

MASKED-LOSS VARIANT: the 41 USA-involving dyads (40 USA<->MENA + the
zero-padded USA<->USA self-dyad) still flow through the whole model - they're
still tokens the cross-dyad attention can look at, so other dyads can use
whatever signal USA's relations carry as CONTEXT. But the loss and all
reported metrics only grade the model on the original 400 MENA<->MENA
dyads - the optimizer never gets gradient pressure to fit USA's own
(possibly noisier / differently-distributed) outcomes. This tests whether
USA was hurting the earlier full-441 model by being an extra, harder
prediction target, rather than by being present at all.

Run:
    python train_goldstein_structural_masked.py
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score
from tensorflow import keras

DATA_PATH = "raw data/bq_panel_21_with_target.parquet"
TARGET_COL = "target_avg_goldstein_next13w"
EXCLUDE_COLS = {"week_start", "Actor1CountryCode", "Actor2CountryCode",
                "action_lat_mean", "action_long_mean", TARGET_COL}

MENA = {
    "ARE", "EGY", "IRN", "IRQ", "ISR", "JOR", "LBY", "PSE", "SAU", "SDN",
    "SYR", "TUR", "YEM", "BHR", "CYP", "KWT", "LBN", "OMN", "QAT", "DJI",
}

N_DYADS_PER_WEEK = 441
LOOKBACK_WEEKS = 13
COUNTRY_EMBED_DIM = 8
D_MODEL = 64
NUM_HEADS = 4
FF_DIM = 128
NUM_TEMPORAL_BLOCKS = 2
NUM_CROSSDYAD_BLOCKS = 2
TRAIN_FRAC = 0.70
VAL_FRAC = 0.10

RELATION_NAMES = ["self", "reverse", "same_source", "same_target",
                   "forward_chain", "backward_chain", "unrelated"]
N_RELATIONS = len(RELATION_NAMES)


def classify_relation(a1_i, a2_i, a1_j, a2_j):
    if a1_i == a1_j and a2_i == a2_j:
        return 0  # self
    if a1_i == a2_j and a2_i == a1_j:
        return 1  # reverse
    if a1_i == a1_j:
        return 2  # same_source
    if a2_i == a2_j:
        return 3  # same_target
    if a2_i == a1_j:
        return 4  # forward_chain
    if a1_i == a2_j:
        return 5  # backward_chain
    return 6  # unrelated


def build_relation_matrix(actor1_per_dyad, actor2_per_dyad):
    n = len(actor1_per_dyad)
    mat = np.zeros((n, n), dtype="int32")
    for i in range(n):
        for j in range(n):
            mat[i, j] = classify_relation(
                actor1_per_dyad[i], actor2_per_dyad[i], actor1_per_dyad[j], actor2_per_dyad[j]
            )
    return mat


class StructuralBiasAttention(keras.layers.Layer):
    """Multi-head self-attention with a learned per-relation-type additive
    bias on the attention logits, in place of Keras's built-in
    MultiHeadAttention (which only supports boolean masks, not an arbitrary
    learned bias)."""

    def __init__(self, num_heads, key_dim, relation_matrix, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.dropout_rate = dropout
        self.relation_matrix = tf.constant(relation_matrix, dtype=tf.int32)  # (N, N), fixed

    def build(self, input_shape):
        d_model = input_shape[-1]
        proj_dim = self.num_heads * self.key_dim
        init = keras.initializers.GlorotUniform()
        self.wq = self.add_weight(shape=(d_model, proj_dim), initializer=init, name="wq")
        self.wk = self.add_weight(shape=(d_model, proj_dim), initializer=init, name="wk")
        self.wv = self.add_weight(shape=(d_model, proj_dim), initializer=init, name="wv")
        self.wo = self.add_weight(shape=(proj_dim, d_model), initializer=init, name="wo")
        # one learned scalar per (relation type, head) - starts at 0, i.e. no
        # bias at all until training shows a relation type matters.
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
            return tf.transpose(t, [0, 2, 1, 3])  # (batch, heads, N, key_dim)

        q = split_heads(tf.matmul(x, self.wq))
        k = split_heads(tf.matmul(x, self.wk))
        v = split_heads(tf.matmul(x, self.wv))

        scores = tf.matmul(q, k, transpose_b=True) / tf.sqrt(tf.cast(self.key_dim, tf.float32))

        # (N, N) relation ids -> (N, N, heads) learned bias -> (heads, N, N)
        bias = tf.gather(self.relation_bias, self.relation_matrix)
        bias = tf.transpose(bias, [2, 0, 1])
        scores = scores + bias[tf.newaxis, :, :, :]

        weights = tf.nn.softmax(scores, axis=-1)
        weights = self.dropout(weights, training=training)

        out = tf.matmul(weights, v)  # (batch, heads, N, key_dim)
        out = tf.transpose(out, [0, 2, 1, 3])
        out = tf.reshape(out, [batch, n, self.num_heads * self.key_dim])
        return tf.matmul(out, self.wo)


def temporal_transformer_block(x, name):
    attn_out = keras.layers.MultiHeadAttention(
        num_heads=NUM_HEADS, key_dim=D_MODEL // NUM_HEADS, dropout=0.1, name=f"{name}_mha"
    )(x, x)
    x = keras.layers.LayerNormalization(name=f"{name}_ln1")(x + attn_out)
    ffn_out = keras.layers.Dense(FF_DIM, activation="relu")(x)
    ffn_out = keras.layers.Dense(D_MODEL)(ffn_out)
    ffn_out = keras.layers.Dropout(0.1)(ffn_out)
    x = keras.layers.LayerNormalization(name=f"{name}_ln2")(x + ffn_out)
    return x


def structural_crossdyad_block(x, relation_matrix, name):
    attn_out = StructuralBiasAttention(
        num_heads=NUM_HEADS, key_dim=D_MODEL // NUM_HEADS,
        relation_matrix=relation_matrix, dropout=0.1, name=f"{name}_structattn"
    )(x)
    x = keras.layers.LayerNormalization(name=f"{name}_ln1")(x + attn_out)
    ffn_out = keras.layers.Dense(FF_DIM, activation="relu")(x)
    ffn_out = keras.layers.Dense(D_MODEL)(ffn_out)
    ffn_out = keras.layers.Dropout(0.1)(ffn_out)
    x = keras.layers.LayerNormalization(name=f"{name}_ln2")(x + ffn_out)
    return x


def main():
    df = pd.read_parquet(DATA_PATH)
    df = df.sort_values(["Actor1CountryCode", "Actor2CountryCode", "week_start"]).reset_index(drop=True)
    n_total_weeks = df.week_start.nunique()
    per_dyad_counts = df.groupby(["Actor1CountryCode", "Actor2CountryCode"]).size()
    assert (per_dyad_counts == n_total_weeks).all(), "expected a fully dense panel"
    assert len(per_dyad_counts) == N_DYADS_PER_WEEK
    print(f"{n_total_weeks} total weeks, {len(per_dyad_counts)} dyads")

    feature_cols = [c for c in df.columns if c not in EXCLUDE_COLS]
    log_cols = [c for c in feature_cols if str(df[c].dtype).startswith("Int")]
    raw_cols = [c for c in feature_cols if c not in log_cols]
    print(f"{len(feature_cols)} feature columns ({len(log_cols)} log1p-transformed counts, "
          f"{len(raw_cols)} raw continuous incl. goldstein/tone/shares)")

    country_encoder = LabelEncoder()
    country_encoder.fit(pd.concat([df.Actor1CountryCode, df.Actor2CountryCode]).unique())
    n_countries = len(country_encoder.classes_)

    numeric = np.concatenate([
        np.log1p(df[log_cols].to_numpy(dtype="float32")),
        df[raw_cols].to_numpy(dtype="float32"),
    ], axis=1)
    n_numeric = numeric.shape[1]
    numeric = numeric.reshape(N_DYADS_PER_WEEK, n_total_weeks, n_numeric)
    target_all = df[TARGET_COL].to_numpy(dtype="float32").reshape(N_DYADS_PER_WEEK, n_total_weeks)

    dyad_actor1 = df.Actor1CountryCode.to_numpy().reshape(N_DYADS_PER_WEEK, n_total_weeks)[:, 0]
    dyad_actor2 = df.Actor2CountryCode.to_numpy().reshape(N_DYADS_PER_WEEK, n_total_weeks)[:, 0]
    a1_idx_per_dyad = country_encoder.transform(dyad_actor1).astype("int32")
    a2_idx_per_dyad = country_encoder.transform(dyad_actor2).astype("int32")

    relation_matrix = build_relation_matrix(dyad_actor1, dyad_actor2)
    counts_by_relation = pd.Series(relation_matrix.ravel()).value_counts().sort_index()
    print("Relation-type counts across all dyad pairs:")
    for i, name in enumerate(RELATION_NAMES):
        print(f"  {name:<15} {counts_by_relation.get(i, 0)}")

    # MENA<->MENA mask: True for the original 400 dyads, False for the 41
    # USA-involving ones (40 USA<->MENA + the zero-padded USA<->USA). USA
    # tokens still flow through the model as attention context - this mask
    # only controls what the LOSS and metrics are graded on.
    mena_mask_np = np.array(
        [(a1 in MENA and a2 in MENA) for a1, a2 in zip(dyad_actor1, dyad_actor2)],
        dtype="float32",
    )
    print(f"Loss/metrics computed on {int(mena_mask_np.sum())} of {N_DYADS_PER_WEEK} dyads "
          f"(USA-involving dyads are context-only, not graded)")
    mena_mask_tf = tf.constant(mena_mask_np)

    def masked_mse(y_true, y_pred):
        sq_err = tf.square(y_pred - y_true) * mena_mask_tf[tf.newaxis, :]
        return tf.reduce_sum(sq_err, axis=-1) / tf.reduce_sum(mena_mask_tf)

    def masked_mae(y_true, y_pred):
        abs_err = tf.abs(y_pred - y_true) * mena_mask_tf[tf.newaxis, :]
        return tf.reduce_sum(abs_err, axis=-1) / tf.reduce_sum(mena_mask_tf)

    windows = sliding_window_view(numeric, window_shape=LOOKBACK_WEEKS, axis=1)
    windows = np.moveaxis(windows, -1, 2)  # (dyads, n_total_weeks-W+1, W, n_numeric)

    n_valid_target = int((~np.isnan(target_all[0])).sum())
    n_pred_weeks = n_valid_target - LOOKBACK_WEEKS + 1
    windows_valid = windows[:, :n_pred_weeks]
    y_valid = target_all[:, LOOKBACK_WEEKS - 1: LOOKBACK_WEEKS - 1 + n_pred_weeks]
    print(f"{n_valid_target} weeks have a valid target; {n_pred_weeks} usable prediction weeks")

    numeric_seq = np.transpose(windows_valid, (1, 0, 2, 3))  # (n_pred_weeks, dyads, W, n_numeric)
    y_all = np.transpose(y_valid, (1, 0))

    n_train = int(n_pred_weeks * TRAIN_FRAC)
    n_val = int(n_pred_weeks * VAL_FRAC)
    n_test = n_pred_weeks - n_train - n_val
    print(f"Split: train={n_train}, val={n_val}, test={n_test} weeks "
          f"(each week = {N_DYADS_PER_WEEK} dyad predictions)")

    def split(arr):
        return arr[:n_train], arr[n_train:n_train + n_val], arr[n_train + n_val:]

    num_train, num_val, num_test = split(numeric_seq)
    y_train, y_val, y_test = split(y_all)

    scaler = StandardScaler()
    scaler.fit(num_train.reshape(-1, n_numeric))
    num_train = scaler.transform(num_train.reshape(-1, n_numeric)).reshape(num_train.shape).astype("float32")
    num_val = scaler.transform(num_val.reshape(-1, n_numeric)).reshape(num_val.shape).astype("float32")
    num_test = scaler.transform(num_test.reshape(-1, n_numeric)).reshape(num_test.shape).astype("float32")

    a1_train = np.tile(a1_idx_per_dyad, (n_train, 1))
    a1_val = np.tile(a1_idx_per_dyad, (n_val, 1))
    a1_test = np.tile(a1_idx_per_dyad, (n_test, 1))
    a2_train = np.tile(a2_idx_per_dyad, (n_train, 1))
    a2_val = np.tile(a2_idx_per_dyad, (n_val, 1))
    a2_test = np.tile(a2_idx_per_dyad, (n_test, 1))

    # ---------------- Model ----------------
    numeric_seq_input = keras.layers.Input(
        shape=(N_DYADS_PER_WEEK, LOOKBACK_WEEKS, n_numeric), name="numeric_seq"
    )
    actor1_input = keras.layers.Input(shape=(N_DYADS_PER_WEEK,), dtype="int32", name="actor1_idx")
    actor2_input = keras.layers.Input(shape=(N_DYADS_PER_WEEK,), dtype="int32", name="actor2_idx")

    token = keras.layers.Dense(D_MODEL, activation="relu", name="token_projection")(numeric_seq_input)

    positions = keras.layers.Lambda(
        lambda t: tf.range(LOOKBACK_WEEKS), output_shape=(LOOKBACK_WEEKS,)
    )(numeric_seq_input)
    temporal_pos_embedding = keras.layers.Embedding(
        input_dim=LOOKBACK_WEEKS, output_dim=D_MODEL, name="temporal_position_embedding"
    )(positions)
    token = keras.layers.Lambda(
        lambda t: t[0] + t[1], name="add_temporal_position"
    )([token, temporal_pos_embedding])

    merged = keras.layers.Lambda(
        lambda t: tf.reshape(t, [-1, LOOKBACK_WEEKS, D_MODEL]), name="merge_batch_dyad"
    )(token)

    x = merged
    for i in range(NUM_TEMPORAL_BLOCKS):
        x = temporal_transformer_block(x, name=f"temporal_block{i}")

    temporal_summary = keras.layers.Lambda(lambda t: t[:, -1, :], name="take_last_week")(x)
    temporal_summary = keras.layers.Lambda(
        lambda t: tf.reshape(t, [-1, N_DYADS_PER_WEEK, D_MODEL]), name="split_batch_dyad"
    )(temporal_summary)

    country_embedding = keras.layers.Embedding(
        input_dim=n_countries, output_dim=COUNTRY_EMBED_DIM, name="country_embedding"
    )
    actor1_vec = country_embedding(actor1_input)
    actor2_vec = country_embedding(actor2_input)

    dyad_token = keras.layers.Concatenate(axis=-1)([temporal_summary, actor1_vec, actor2_vec])
    dyad_token = keras.layers.Dense(D_MODEL, activation="relu", name="dyad_token_projection")(dyad_token)

    y = dyad_token
    for i in range(NUM_CROSSDYAD_BLOCKS):
        y = structural_crossdyad_block(y, relation_matrix, name=f"crossdyad_block{i}")

    output = keras.layers.Dense(1, name="per_dyad_output")(y)
    output = keras.layers.Reshape((N_DYADS_PER_WEEK,), name="squeeze")(output)

    model = keras.Model(inputs=[numeric_seq_input, actor1_input, actor2_input], outputs=output)
    model.compile(optimizer="adam", loss=masked_mse, metrics=[masked_mae])
    model.summary()

    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=8, restore_best_weights=True
    )

    model.fit(
        {"numeric_seq": num_train, "actor1_idx": a1_train, "actor2_idx": a2_train}, y_train,
        validation_data=(
            {"numeric_seq": num_val, "actor1_idx": a1_val, "actor2_idx": a2_val}, y_val
        ),
        epochs=100,
        batch_size=8,
        callbacks=[early_stop],
    )

    y_pred = model.predict(
        {"numeric_seq": num_test, "actor1_idx": a1_test, "actor2_idx": a2_test}, verbose=0
    )

    mena_bool = mena_mask_np.astype(bool)
    # Headline numbers: MENA<->MENA only (the 400 dyads actually graded during training)
    y_pred_mena = y_pred[:, mena_bool].ravel()
    y_test_mena = y_test[:, mena_bool].ravel()

    test_mse = np.mean((y_pred_mena - y_test_mena) ** 2)
    test_mae = mean_absolute_error(y_test_mena, y_pred_mena)
    test_r2 = r2_score(y_test_mena, y_pred_mena)

    train_mena_mean = y_train[:, mena_bool].ravel().mean()
    baseline_pred = np.full_like(y_test_mena, train_mena_mean)
    baseline_mse = np.mean((baseline_pred - y_test_mena) ** 2)
    baseline_mae = mean_absolute_error(y_test_mena, baseline_pred)

    print(f"\n=== MENA<->MENA dyads only (400 of 441 - what the loss actually graded) ===")
    print(f"Test MSE:  {test_mse:.4f}   (always-predict-train-mean baseline: {baseline_mse:.4f})")
    print(f"Test MAE:  {test_mae:.4f}   (baseline: {baseline_mae:.4f})")
    print(f"Test R^2:  {test_r2:.4f}   (0 = no better than predicting the mean, 1 = perfect)")

    # For reference only: how the (never-graded) USA dyads happen to look,
    # purely from weights shaped by the MENA<->MENA loss.
    y_pred_usa = y_pred[:, ~mena_bool].ravel()
    y_test_usa = y_test[:, ~mena_bool].ravel()
    usa_mae = mean_absolute_error(y_test_usa, y_pred_usa)
    usa_r2 = r2_score(y_test_usa, y_pred_usa)
    print(f"\n(For reference, not graded during training) USA-involving dyads: "
          f"MAE={usa_mae:.4f}, R^2={usa_r2:.4f}")

    # Inspect what the model learned about each structural relationship
    print("\nLearned relation biases (averaged across heads), per cross-dyad block:")
    for layer in model.layers:
        if isinstance(layer, StructuralBiasAttention):
            w = layer.relation_bias.numpy().mean(axis=1)
            print(f"  {layer.name}:")
            for name, val in zip(RELATION_NAMES, w):
                print(f"    {name:<15} {val:+.4f}")

    model.save_weights("model_goldstein_structural_masked.weights.h5")
    print("\nSaved trained weights to model_goldstein_structural_masked.weights.h5")


if __name__ == "__main__":
    main()
