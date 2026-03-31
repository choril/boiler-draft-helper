"""
Transformer 多目标多步预测模型

特点:
1. Transformer Encoder - 强大的时序特征提取
2. Transformer Decoder - 自回归多步预测
3. 多目标输出头 - 负压和含氧量独立预测
4. 物理耦合层 - 学习目标间关系
5. 位置编码 - 区分时间步和预测步
"""

import json
import os
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np
import tensorflow as tf

tf.get_logger().setLevel("ERROR")

from tensorflow import keras
from tensorflow.keras import layers, callbacks, optimizers

from src.utils.utils import print_section


def combined_loss(y_true, y_pred):
    """组合损失: 负压用Huber, 含氧量用MSE"""
    y_true_pressure = y_true[:, :, 0:1]
    y_true_oxygen = y_true[:, :, 1:2]
    y_pred_pressure = y_pred[:, :, 0:1]
    y_pred_oxygen = y_pred[:, :, 1:2]

    huber = tf.keras.losses.Huber(delta=1.0, reduction='sum_over_batch_size')
    pressure_loss = huber(y_true_pressure, y_pred_pressure)

    mse = tf.keras.losses.MeanSquaredError(reduction='sum_over_batch_size')
    oxygen_loss = mse(y_true_oxygen, y_pred_oxygen)

    return pressure_loss + oxygen_loss * 0.5


def setup_gpu(gpus: str = "all", memory_growth: bool = True) -> tf.distribute.Strategy:
    """配置GPU"""
    gpus_list = tf.config.list_physical_devices("GPU")

    if not gpus_list:
        print("未检测到GPU，使用CPU训练")
        return tf.distribute.get_strategy()

    if gpus != "all":
        gpu_ids = [int(g.strip()) for g in gpus.split(",")]
        gpus_list = [gpus_list[i] for i in gpu_ids if i < len(gpus_list)]

    tf.config.set_visible_devices(gpus_list, "GPU")

    if memory_growth:
        for gpu in gpus_list:
            tf.config.experimental.set_memory_growth(gpu, True)

    print(f"可用GPU: {len(gpus_list)} 个")
    for i, gpu in enumerate(gpus_list):
        print(f"  - {gpu.name}")

    if len(gpus_list) > 1:
        devices = [f"/GPU:{i}" for i in range(len(gpus_list))]
        print(f"使用 MirroredStrategy 多GPU训练 (NCCL AllReduce)")
        strategy = tf.distribute.MirroredStrategy(
            devices=devices,
            cross_device_ops=tf.distribute.NcclAllReduce()
        )
    else:
        strategy = tf.distribute.get_strategy()
        print(f"使用单GPU训练")

    return strategy


class PositionalEncoding(layers.Layer):
    """正弦位置编码"""

    def __init__(self, d_model: int, max_len: int = 5000, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.max_len = max_len

    def build(self, input_shape):
        # 预计算位置编码
        position = np.arange(self.max_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, self.d_model, 2) * -(np.log(10000.0) / self.d_model))

        pe = np.zeros((self.max_len, self.d_model))
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)

        self.pe = tf.constant(pe[np.newaxis, :, :], dtype=tf.float32)

    def call(self, inputs):
        seq_len = tf.shape(inputs)[1]
        return inputs + self.pe[:, :seq_len, :]

    def get_config(self):
        config = super().get_config()
        config.update({"d_model": self.d_model, "max_len": self.max_len})
        return config


class LearnablePositionalEncoding(layers.Layer):
    """可学习位置编码 - 用于输出序列"""

    def __init__(self, d_model: int, max_steps: int = 10, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.max_steps = max_steps

    def build(self, input_shape):
        self.pos_embedding = self.add_weight(
            shape=(self.max_steps, self.d_model),
            initializer="glorot_uniform",
            trainable=True,
            name="pos_embedding"
        )

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        seq_len = tf.shape(inputs)[1]

        pos_encoding = tf.expand_dims(self.pos_embedding[:seq_len, :], axis=0)
        pos_encoding = tf.tile(pos_encoding, [batch_size, 1, 1])

        return inputs + pos_encoding

    def get_config(self):
        config = super().get_config()
        config.update({"d_model": self.d_model, "max_steps": self.max_steps})
        return config


class MultiHeadSelfAttention(layers.Layer):
    """多头自注意力"""

    def __init__(self, d_model: int, num_heads: int = 8, dropout: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.dropout_rate = dropout

        assert d_model % num_heads == 0, "d_model 必须能被 num_heads 整除"
        self.depth = d_model // num_heads

    def build(self, input_shape):
        self.wq = layers.Dense(self.d_model, name="query")
        self.wk = layers.Dense(self.d_model, name="key")
        self.wv = layers.Dense(self.d_model, name="value")
        self.dropout = layers.Dropout(self.dropout_rate)
        self.projection = layers.Dense(self.d_model, name="projection")

    def call(self, inputs, mask=None, training=False):
        batch_size = tf.shape(inputs)[0]

        # Linear projections
        q = self.wq(inputs)  # (batch, seq_len, d_model)
        k = self.wk(inputs)
        v = self.wv(inputs)

        # Split into multiple heads
        q = self._split_heads(q, batch_size)
        k = self._split_heads(k, batch_size)
        v = self._split_heads(v, batch_size)

        # Scaled dot-product attention
        attention_scores = tf.matmul(q, k, transpose_b=True)
        attention_scores = attention_scores / tf.math.sqrt(tf.cast(self.depth, tf.float32))

        if mask is not None:
            attention_scores += (mask * -1e9)

        attention_weights = tf.nn.softmax(attention_scores, axis=-1)
        attention_weights = self.dropout(attention_weights, training=training)

        # Apply attention to values
        output = tf.matmul(attention_weights, v)
        output = self._combine_heads(output, batch_size)

        return self.projection(output), attention_weights

    def _split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth)"""
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def _combine_heads(self, x, batch_size):
        """Combine heads back to single dimension"""
        x = tf.transpose(x, perm=[0, 2, 1, 3])
        return tf.reshape(x, (batch_size, -1, self.d_model))

    def get_config(self):
        config = super().get_config()
        config.update({
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "dropout": self.dropout_rate
        })
        return config


class TransformerEncoderLayer(layers.Layer):
    """Transformer Encoder Layer"""

    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        dff: int = 512,
        dropout: float = 0.1,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.dropout_rate = dropout

    def build(self, input_shape):
        self.mha = MultiHeadSelfAttention(self.d_model, self.num_heads, self.dropout_rate)
        self.ffn = keras.Sequential([
            layers.Dense(self.dff, activation="relu"),
            layers.Dense(self.d_model)
        ], name="feed_forward")

        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(self.dropout_rate)
        self.dropout2 = layers.Dropout(self.dropout_rate)

    def call(self, inputs, mask=None, training=False):
        # Multi-head attention with residual connection
        attn_output, _ = self.mha(inputs, mask=mask, training=training)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)

        # Feed forward with residual connection
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)

        return out2

    def get_config(self):
        config = super().get_config()
        config.update({
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "dff": self.dff,
            "dropout": self.dropout_rate
        })
        return config


class TransformerDecoderLayer(layers.Layer):
    """Transformer Decoder Layer with cross-attention"""

    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        dff: int = 512,
        dropout: float = 0.1,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.dropout_rate = dropout

    def build(self, input_shape):
        # Self-attention for decoder
        self.self_mha = MultiHeadSelfAttention(self.d_model, self.num_heads, self.dropout_rate)

        # Cross-attention layers (query from decoder, key/value from encoder)
        self.wq_cross = layers.Dense(self.d_model, name="cross_query")
        self.wk_cross = layers.Dense(self.d_model, name="cross_key")
        self.wv_cross = layers.Dense(self.d_model, name="cross_value")
        self.cross_projection = layers.Dense(self.d_model, name="cross_projection")

        # Feed forward
        self.ffn = keras.Sequential([
            layers.Dense(self.dff, activation="relu"),
            layers.Dense(self.d_model)
        ], name="feed_forward")

        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = layers.Dropout(self.dropout_rate)
        self.dropout2 = layers.Dropout(self.dropout_rate)
        self.dropout3 = layers.Dropout(self.dropout_rate)

        self.depth = self.d_model // self.num_heads

    def _cross_attention(self, query, key, value, training=False):
        """Cross-attention: query from decoder, key/value from encoder"""
        batch_size = tf.shape(query)[0]

        # Linear projections
        q = self.wq_cross(query)
        k = self.wk_cross(key)
        v = self.wv_cross(value)

        # Split heads
        q = self._split_heads(q, batch_size)
        k = self._split_heads(k, batch_size)
        v = self._split_heads(v, batch_size)

        # Scaled dot-product attention
        attention_scores = tf.matmul(q, k, transpose_b=True)
        attention_scores = attention_scores / tf.math.sqrt(tf.cast(self.depth, tf.float32))

        attention_weights = tf.nn.softmax(attention_scores, axis=-1)
        attention_weights = self.dropout2(attention_weights, training=training)

        # Apply attention
        output = tf.matmul(attention_weights, v)
        output = self._combine_heads(output, batch_size)

        return self.cross_projection(output)

    def _split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def _combine_heads(self, x, batch_size):
        x = tf.transpose(x, perm=[0, 2, 1, 3])
        return tf.reshape(x, (batch_size, -1, self.d_model))

    def call(self, inputs, encoder_output, look_ahead_mask=None, padding_mask=None, training=False):
        # Self-attention
        attn1, _ = self.self_mha(inputs, mask=look_ahead_mask, training=training)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(inputs + attn1)

        # Cross-attention to encoder
        attn2 = self._cross_attention(out1, encoder_output, encoder_output, training=training)
        out2 = self.layernorm2(out1 + attn2)

        # Feed forward
        ffn_output = self.ffn(out2)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(out2 + ffn_output)

        return out3

    def get_config(self):
        config = super().get_config()
        config.update({
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "dff": self.dff,
            "dropout": self.dropout_rate
        })
        return config


class PhysicsCouplingLayer(layers.Layer):
    """物理耦合层 - 学习负压和含氧量之间的关系"""

    def __init__(self, units: int = 32, **kwargs):
        super().__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        self.shared = layers.Dense(self.units, activation="relu")
        self.pressure_to_oxygen = layers.Dense(self.units, activation="sigmoid")
        self.oxygen_to_pressure = layers.Dense(self.units, activation="sigmoid")
        self.output_proj = layers.Dense(1)
        self.built = True

    def call(self, inputs):
        pressure_features, oxygen_features = inputs

        shared_pressure = self.shared(pressure_features)
        shared_oxygen = self.shared(oxygen_features)

        pressure_influence = self.pressure_to_oxygen(shared_pressure)
        oxygen_influence = self.oxygen_to_pressure(shared_oxygen)

        pressure_combined = pressure_features + self.output_proj(oxygen_influence * shared_oxygen)
        oxygen_combined = oxygen_features + self.output_proj(pressure_influence * shared_pressure)

        return pressure_combined, oxygen_combined

    def get_config(self):
        config = super().get_config()
        config.update({"units": self.units})
        return config


class MultiStepTransformer:
    """
    Transformer 多目标多步预测模型

    架构:
    1. Input Embedding: 特征投影到 d_model 维度
    2. Positional Encoding: 输入序列位置编码
    3. Encoder Stack: N 层 Transformer Encoder
    4. Decoder Input: 可学习的目标序列嵌入 + 位置编码
    5. Decoder Stack: N 层 Transformer Decoder
    6. Output Heads: 负压和含氧量独立预测头 + 物理耦合
    """

    def __init__(
        self,
        seq_length: int = 30,
        n_features: int = 79,
        output_steps: int = 10,
        d_model: int = 128,
        num_heads: int = 8,
        dff: int = 512,
        num_encoder_layers: int = 4,
        num_decoder_layers: int = 4,
        dropout_rate: float = 0.1,
        learning_rate: float = 0.001,
        strategy: tf.distribute.Strategy | None = None,
        feature_names: list[str] | None = None,
    ):
        self.seq_length = seq_length
        self.n_features = n_features
        self.output_steps = output_steps
        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.strategy = strategy or tf.distribute.get_strategy()
        self.feature_names = feature_names or [f"feature_{i}" for i in range(n_features)]

        self.model: keras.Model | None = None
        self.history: callbacks.History | None = None

        self._build_model()

    def _build_model(self) -> None:
        """构建完整的 Transformer 模型"""
        with self.strategy.scope():
            # ==================== Encoder ====================
            encoder_inputs = layers.Input(
                shape=(self.seq_length, self.n_features),
                name="encoder_inputs"
            )

            # Input embedding - 投影到 d_model 维度
            x = layers.Dense(self.d_model, name="input_embedding")(encoder_inputs)

            # Positional encoding for input sequence
            x = PositionalEncoding(self.d_model, max_len=self.seq_length, name="input_pos_encoding")(x)
            x = layers.Dropout(self.dropout_rate, name="input_dropout")(x)

            # Encoder stack
            encoder_output = x
            for i in range(self.num_encoder_layers):
                encoder_output = TransformerEncoderLayer(
                    d_model=self.d_model,
                    num_heads=self.num_heads,
                    dff=self.dff,
                    dropout=self.dropout_rate,
                    name=f"encoder_layer_{i}"
                )(encoder_output)

            # ==================== Decoder ====================
            # Learnable decoder input - 每个预测步一个可学习的向量
            decoder_input = layers.Input(
                shape=(self.output_steps, self.d_model),
                name="decoder_input"
            )

            # Add learnable positional encoding for output steps
            decoder_seq = LearnablePositionalEncoding(
                self.d_model,
                max_steps=self.output_steps,
                name="decoder_pos_encoding"
            )(decoder_input)

            decoder_seq = layers.Dropout(self.dropout_rate, name="decoder_dropout")(decoder_seq)

            # Decoder stack
            decoder_output = decoder_seq
            for i in range(self.num_decoder_layers):
                decoder_output = TransformerDecoderLayer(
                    d_model=self.d_model,
                    num_heads=self.num_heads,
                    dff=self.dff,
                    dropout=self.dropout_rate,
                    name=f"decoder_layer_{i}"
                )(decoder_output, encoder_output)

            # ==================== Output Heads ====================
            # Shared feature extraction
            shared = layers.Dense(128, activation="relu", name="shared_dense_1")(decoder_output)
            shared = layers.Dropout(self.dropout_rate, name="shared_dropout")(shared)
            shared = layers.Dense(64, activation="relu", name="shared_dense_2")(shared)

            # Pressure head
            pressure = layers.Dense(32, activation="relu", name="pressure_hidden")(shared)
            pressure = layers.Dropout(self.dropout_rate)(pressure)
            pressure_out = layers.Dense(1, name="pressure_raw")(pressure)

            # Oxygen head
            oxygen = layers.Dense(32, activation="relu", name="oxygen_hidden")(shared)
            oxygen = layers.Dropout(self.dropout_rate)(oxygen)
            oxygen_out = layers.Dense(1, name="oxygen_raw")(oxygen)

            # Physics coupling
            pressure_final, oxygen_final = PhysicsCouplingLayer(16, name="physics_coupling")(
                [pressure_out, oxygen_out]
            )

            # Combine outputs
            outputs = layers.Concatenate(name="output")([pressure_final, oxygen_final])

            # Create model
            self.model = keras.Model(
                inputs=[encoder_inputs, decoder_input],
                outputs=outputs,
                name="MultiStepTransformer"
            )

            # Create decoder input tensor (learnable)
            self.decoder_input_init = tf.Variable(
                tf.random.normal([1, self.output_steps, self.d_model]),
                trainable=True,
                name="decoder_input_embedding"
            )

            # Compile
            self.model.compile(
                optimizer=optimizers.Adam(learning_rate=self.learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9),
                loss=combined_loss,
                metrics=["mae"]
            )

    def _create_decoder_input(self, batch_size: int) -> tf.Tensor:
        """创建解码器输入"""
        return tf.tile(self.decoder_input_init, [batch_size, 1, 1])

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        epochs: int = 100,
        batch_size: int = 32,
        patience: int = 15,
        verbose: int = 1,
    ) -> callbacks.History:
        """训练模型"""
        print_section("Transformer 多步预测模型训练")
        print(f"样本: {X_train.shape[0]}, 序列: {X_train.shape[1]}, 特征: {X_train.shape[2]}")
        print(f"输出步数: {self.output_steps}")
        print(f"d_model: {self.d_model}, heads: {self.num_heads}, layers: {self.num_encoder_layers}/{self.num_decoder_layers}")

        n_gpus = self.strategy.num_replicas_in_sync
        if n_gpus > 1:
            batch_size = batch_size * n_gpus
            print(f"多GPU训练: {n_gpus} GPUs, 有效batch_size: {batch_size}")

        # Create decoder inputs
        decoder_input_train = np.tile(
            self.decoder_input_init.numpy(),
            (X_train.shape[0], 1, 1)
        )

        train_dataset = tf.data.Dataset.from_tensor_slices(
            ((X_train, decoder_input_train), y_train)
        )
        train_dataset = train_dataset.shuffle(10000).batch(batch_size, drop_remainder=True if n_gpus > 1 else False)
        train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)

        val_dataset = None
        if X_val is not None and y_val is not None:
            decoder_input_val = np.tile(
                self.decoder_input_init.numpy(),
                (X_val.shape[0], 1, 1)
            )
            val_dataset = tf.data.Dataset.from_tensor_slices(
                ((X_val, decoder_input_val), y_val)
            )
            val_dataset = val_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

        # Callbacks
        cb = [
            callbacks.EarlyStopping(
                monitor="val_loss" if val_dataset else "loss",
                patience=patience,
                restore_best_weights=True,
                verbose=1
            ),
            callbacks.ReduceLROnPlateau(
                monitor="val_loss" if val_dataset else "loss",
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=1
            ),
            callbacks.ModelCheckpoint(
                filepath="output/deep_learning/best_model_transformer.keras",
                monitor="val_loss" if val_dataset else "loss",
                save_best_only=True,
                verbose=1
            ),
        ]

        self.history = self.model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=epochs,
            callbacks=cb,
            verbose=verbose
        )

        return self.history

    def predict(self, X: np.ndarray, batch_size: int | None = None) -> np.ndarray:
        """预测"""
        if batch_size is None:
            batch_size = 128 * self.strategy.num_replicas_in_sync

        decoder_input = np.tile(
            self.decoder_input_init.numpy(),
            (X.shape[0], 1, 1)
        )

        return self.model.predict((X, decoder_input), batch_size=batch_size, verbose=0)

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        """评估模型"""
        print_section("模型评估")

        pred = self.predict(X_test)

        results = {}
        print("\n各步预测评估:")
        print("-" * 60)
        print(f"{'Step':<8} {'MAE(负压)':<12} {'MAE(含氧量)':<12} {'RMSE':<12}")
        print("-" * 60)

        for s in range(1, self.output_steps + 1):
            pred_s = pred[:, s - 1, :]
            true_s = y_test[:, s - 1, :]

            mae_per_target = np.mean(np.abs(true_s - pred_s), axis=0)
            rmse = np.sqrt(np.mean((true_s - pred_s) ** 2))

            results[str(s)] = {
                "mae": float(np.mean(mae_per_target)),
                "rmse": float(rmse),
                "mae_pressure": float(mae_per_target[0]),
                "mae_oxygen": float(mae_per_target[1]),
            }
            print(f"{s:<8} {mae_per_target[0]:<12.4f} {mae_per_target[1]:<12.4f} {rmse:<12.4f}")

        print("-" * 60)
        return results

    def summary(self) -> None:
        """打印模型结构"""
        self.model.summary()

    def save(self, path: str | Path) -> None:
        """保存模型"""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save model weights - Keras requires .weights.h5 extension
        self.model.save_weights(path / "model.weights.h5")

        # Save config
        config = {
            "seq_length": self.seq_length,
            "n_features": self.n_features,
            "output_steps": self.output_steps,
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "dff": self.dff,
            "num_encoder_layers": self.num_encoder_layers,
            "num_decoder_layers": self.num_decoder_layers,
            "dropout_rate": self.dropout_rate,
            "learning_rate": self.learning_rate,
        }
        with open(path / "config.json", "w") as f:
            json.dump(config, f, indent=2)

        # Save decoder input
        np.save(path / "decoder_input.npy", self.decoder_input_init.numpy())

        print(f"模型已保存: {path}")

    @classmethod
    def load(cls, path: str | Path, strategy: tf.distribute.Strategy | None = None) -> "MultiStepTransformer":
        """加载模型"""
        path = Path(path)

        with open(path / "config.json", "r") as f:
            config = json.load(f)

        instance = cls(**config, strategy=strategy)

        # Load decoder input
        decoder_input = np.load(path / "decoder_input.npy")
        instance.decoder_input_init.assign(decoder_input)

        # Load weights - Keras requires .weights.h5 extension
        instance.model.load_weights(path / "model.weights.h5")

        return instance


# ==================== 训练入口 ====================
if __name__ == "__main__":
    import argparse
    import pandas as pd

    parser = argparse.ArgumentParser(description="Transformer 多目标多步预测")
    parser.add_argument("--data", type=str, default="output/all_data_cleaned.feather")
    parser.add_argument("--feature-path", type=str, default="output/features/feature_matrix.feather")
    parser.add_argument("--selection-path", type=str, default="output/features/final_selected_features.json")
    parser.add_argument("--output-dir", type=str, default="output")
    parser.add_argument("--seq-length", type=int, default=30)
    parser.add_argument("--output-steps", type=int, default=10)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--dff", type=int, default=512)
    parser.add_argument("--num-encoder-layers", type=int, default=4)
    parser.add_argument("--num-decoder-layers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--gpus", type=str, default="all")

    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup GPU
    strategy = setup_gpu(args.gpus)

    # Load data
    print_section("加载数据")
    df = pd.read_feather(args.data)
    print(f"数据维度: {df.shape}")

    # Load features
    print_section("加载特征")
    feature_matrix = pd.read_feather(args.feature_path)
    print(f"特征维度: {feature_matrix.shape}")

    # Load selected features
    with open(args.selection_path, "r") as f:
        saved = json.load(f)

    if "selected_features" in saved:
        features = saved["selected_features"]
    else:
        features = []
        for target in saved:
            features.extend(saved[target])
        features = list(dict.fromkeys(features))

    print(f"选中特征数: {len(features)}")

    # Build sequences (simplified)
    from src.features.selector import FeatureSelector
    from src.utils.config import PRESSURE_MAIN, OXYGEN_MAIN

    selector = FeatureSelector(feature_matrix, target_vars=[PRESSURE_MAIN, OXYGEN_MAIN])
    selector.selected_features = features
    selector.fit_scaler(target="targets")

    print_section("构建序列")
    X, y = selector.build_seq2seq_sequences(
        seq_length=args.seq_length,
        output_steps=args.output_steps
    )

    # Split
    n = len(X)
    n_test = int(n * 0.15)
    n_val = int((n - n_test) * 0.15)

    X_test, y_test = X[-n_test:], y[-n_test:]
    X_val, y_val = X[-(n_test + n_val):-n_test], y[-(n_test + n_val):-n_test]
    X_train, y_train = X[:-(n_test + n_val)], y[:-(n_test + n_val)]

    print(f"训练集: {len(X_train)}, 验证集: {len(X_val)}, 测试集: {len(X_test)}")

    # Build and train model
    model = MultiStepTransformer(
        seq_length=args.seq_length,
        n_features=len(features),
        output_steps=args.output_steps,
        d_model=args.d_model,
        num_heads=args.num_heads,
        dff=args.dff,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        dropout_rate=args.dropout,
        strategy=strategy,
        feature_names=features,
    )

    model.summary()

    model.train(
        X_train, y_train,
        X_val, y_val,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )

    # Evaluate
    results = model.evaluate(X_test, y_test)

    # Print denormalized results
    if selector.target_scaler is not None:
        print("\n反标准化后的评估指标:")
        print("-" * 70)
        print(f"{'Step':<8} {'MAE负压(Pa)':<15} {'MAE含氧量(%)':<15} {'RMSE':<12}")
        print("-" * 70)

        for s, r in results.items():
            mae_pressure_orig = r["mae_pressure"] * selector.target_scaler.scale_[0]
            mae_oxygen_orig = r["mae_oxygen"] * selector.target_scaler.scale_[1]
            rmse_orig = r["rmse"] * np.mean(selector.target_scaler.scale_)
            print(f"{s:<8} {mae_pressure_orig:<15.2f} {mae_oxygen_orig:<15.4f} {rmse_orig:<12.2f}")

        print("-" * 70)

    # Save
    model.save(output_dir / "models" / "transformer" / "model")

    # Save results
    results_path = output_dir / "models" / "transformer" / "evaluation_results.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print("\n" + "=" * 70)
    print("训练完成!")
    print("=" * 70)