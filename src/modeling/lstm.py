"""
LSTM模型 - Query-based 多步预测架构

架构：
1. 单层Bi-LSTM编码器（避免过拟合）
2. Query-based解码器（可学习查询向量 + Cross-Attention）
3. 物理约束损失（平滑性）
4. 增强的物理耦合层
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

from src.utils.utils import physics_guided_loss
from src.utils.logger import get_logger
logger = get_logger(__name__)

# =============================================================================
# 物理耦合层
# =============================================================================

class PhysicsCouplingLayer(layers.Layer):
    """物理耦合层 - 建模负压和含氧量的非线性交互"""

    def __init__(self, units: int = 32, **kwargs):
        super().__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        self.shared = layers.Dense(self.units, activation="relu", name="shared")
        self.p_gate = layers.Dense(self.units, activation="sigmoid", name="p_gate")
        self.o_gate = layers.Dense(self.units, activation="sigmoid", name="o_gate")
        self.p_proj = layers.Dense(1, name="p_proj")
        self.o_proj = layers.Dense(1, name="o_proj")

    def call(self, inputs):
        p_feat, o_feat = inputs  # (batch, time, 1)

        # 共享特征
        p_hidden = self.shared(p_feat)
        o_hidden = self.shared(o_feat)

        # 门控交互
        p_to_o = self.o_gate(p_hidden) * o_hidden
        o_to_p = self.p_gate(o_hidden) * p_hidden

        # 融合输出（残差连接，小系数防止过拟合）
        p_out = p_feat + self.p_proj(o_to_p) * 0.1
        o_out = o_feat + self.o_proj(p_to_o) * 0.1

        return p_out, o_out

    def get_config(self):
        config = super().get_config()
        config.update({"units": self.units})
        return config


class PredictStepEmbedding(layers.Layer):
    """可学习的预测步嵌入 - 为每个预测步提供唯一的查询向量"""

    def __init__(self, d_model: int, max_steps: int = 10, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.max_steps = max_steps

    def build(self, input_shape):
        self.query_embeddings = self.add_weight(
            shape=(self.max_steps, self.d_model),
            initializer="glorot_uniform",
            trainable=True,
            name="query_embeddings"
        )

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        queries = tf.expand_dims(self.query_embeddings, axis=0)
        return tf.tile(queries, [batch_size, 1, 1])

    def get_config(self):
        config = super().get_config()
        config.update({"d_model": self.d_model, "max_steps": self.max_steps})
        return config


# =============================================================================
# LSTM模型
# =============================================================================

class LSTM:
    """
    Query-based 多步预测模型

    特点:
    1. 单层Bi-LSTM编码器（减少过拟合）
    2. Query-based解码器（可学习查询向量）
    3. Cross-Attention（从编码器动态提取信息）
    4. 物理约束损失（平滑性）
    """

    def __init__(
        self,
        seq_length: int = 30,
        n_features: int = 50,
        output_steps: int = 10,
        hidden_units: int = 128,
        dropout_rate: float = 0.3,
        learning_rate: float = 0.001,
        l2_reg: float = 1e-3,
        strategy: tf.distribute.Strategy | None = None,
        feature_names: list[str] | None = None,
        smoothness_weight: float = 0.001,
    ):
        self.seq_length = seq_length
        self.n_features = n_features
        self.output_steps = output_steps
        self.hidden_units = hidden_units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.l2_reg = l2_reg
        self.strategy = strategy or tf.distribute.get_strategy()
        self.feature_names = feature_names or [f"feature_{i}" for i in range(n_features)]
        self.smoothness_weight = smoothness_weight
        self.model: keras.Model | None = None
        self.history: callbacks.History | None = None
        self._build_model()

    def _build_model(self) -> None:
        """构建 Query-based 多步预测模型"""
        with self.strategy.scope():
            # 输入
            inputs = layers.Input(
                shape=(self.seq_length, self.n_features),
                name="encoder_input"
            )

            # 特征嵌入（高维时降维）
            if self.n_features > 64:
                x = layers.TimeDistributed(
                    layers.Dense(64, activation="relu"),
                    name="feature_embedding"
                )(inputs)
            else:
                x = inputs

            # 输入投影
            x = layers.Dense(64, activation="relu", name="input_proj")(x)

            # === Bi-LSTM Encoder ===
            encoder_out = layers.Bidirectional(
                layers.LSTM(
                    self.hidden_units,
                    return_sequences=True,
                    kernel_regularizer=keras.regularizers.l2(self.l2_reg),
                    name="encoder_lstm"
                ),
                name="bi_encoder"
            )(x)
            encoder_out = layers.LayerNormalization(name="encoder_norm")(encoder_out)
            encoder_out = layers.Dropout(self.dropout_rate, name="encoder_dropout")(encoder_out)

            # === Query-based Decoder ===
            d_model = self.hidden_units * 2  # Bi-LSTM 输出维度

            # 可学习查询向量
            queries = PredictStepEmbedding(
                d_model=d_model,
                max_steps=self.output_steps,
                name="query_embedding"
            )(encoder_out)

            # Cross-Attention
            attn_out = layers.MultiHeadAttention(
                num_heads=4,
                key_dim=d_model // 4,
                dropout=self.dropout_rate,
                name="cross_attention"
            )(queries, encoder_out)

            decoder_out = layers.LayerNormalization(name="decoder_norm")(queries + attn_out)
            decoder_out = layers.Dropout(self.dropout_rate, name="decoder_dropout")(decoder_out)

            # Feed-forward
            ff = layers.Dense(d_model * 2, activation="gelu", name="ff_dense1")(decoder_out)
            ff = layers.Dropout(self.dropout_rate, name="ff_dropout")(ff)
            ff = layers.Dense(d_model, name="ff_dense2")(ff)
            decoder_out = layers.LayerNormalization(name="ff_norm")(decoder_out + ff)

            # === 分离预测头 ===
            # 负压分支（输出头使用较低dropout）
            p_hidden = layers.TimeDistributed(
                layers.Dense(64, activation="relu"),
                name="pressure_hidden"
            )(decoder_out)
            p_hidden = layers.TimeDistributed(
                layers.Dropout(self.dropout_rate * 0.5),
                name="pressure_dropout"
            )(p_hidden)
            p_out = layers.TimeDistributed(layers.Dense(1), name="pressure_raw")(p_hidden)

            # 含氧量分支（输出头使用较低dropout）
            o_hidden = layers.TimeDistributed(
                layers.Dense(64, activation="relu"),
                name="oxygen_hidden"
            )(decoder_out)
            o_hidden = layers.TimeDistributed(
                layers.Dropout(self.dropout_rate * 0.5),
                name="oxygen_dropout"
            )(o_hidden)
            o_out = layers.TimeDistributed(layers.Dense(1), name="oxygen_raw")(o_hidden)

            # 物理耦合
            p_final, o_final = PhysicsCouplingLayer(32, name="physics_coupling")([p_out, o_out])

            # 合并输出
            outputs = layers.Concatenate(name="output")([p_final, o_final])

            self.model = keras.Model(inputs=inputs, outputs=outputs, name="LSTM")

            # 定义损失函数（闭包传递 smoothness_weight）
            def loss_fn(y_true, y_pred):
                return physics_guided_loss(y_true, y_pred, self.smoothness_weight)

            self.model.compile(
                optimizer=optimizers.Adam(learning_rate=self.learning_rate, clipnorm=1.0),
                loss=loss_fn,
                metrics=["mae"],
            )

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        epochs: int = 100,
        batch_size: int = 64,
        patience: int = 15,
        verbose: int = 1,
    ) -> callbacks.History:
        logger.info("模型训练")
        logger.info(f"样本: {X_train.shape[0]}, 序列长度: {X_train.shape[1]}, 特征: {X_train.shape[2]}")
        logger.info(f"输出步数: {self.output_steps}")

        n_gpus = self.strategy.num_replicas_in_sync
        if n_gpus > 1:
            batch_size = batch_size * n_gpus
            logger.info(f"多GPU训练: {n_gpus} GPUs, 有效batch_size: {batch_size}")

        def make_dataset(X, y, batch_size, shuffle=False):
            ds = tf.data.Dataset.from_tensor_slices((X, y))
            if not shuffle:
                ds = ds.cache()
            if shuffle:
                ds = ds.shuffle(buffer_size=10000, reshuffle_each_iteration=True)
            ds = ds.batch(batch_size, drop_remainder=True if n_gpus > 1 else False)
            ds = ds.prefetch(tf.data.AUTOTUNE)
            return ds

        train_dataset = make_dataset(X_train, y_train, batch_size, shuffle=True)

        val_dataset = None
        if X_val is not None and y_val is not None:
            val_dataset = make_dataset(X_val, y_val, batch_size, shuffle=False)

        cb = [
            callbacks.EarlyStopping(
                monitor="val_loss" if val_dataset else "loss",
                patience=patience,
                restore_best_weights=True,
                verbose=1,
            ),
            callbacks.ReduceLROnPlateau(
                monitor="val_loss" if val_dataset else "loss",
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=1,
            ),
            callbacks.ModelCheckpoint(
                filepath="output/models/lstm/best_model_lstm.keras",
                monitor="val_loss" if val_dataset else "loss",
                save_best_only=True,
                verbose=1,
            ),
        ]

        self.history = self.model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=epochs,
            callbacks=cb,
            verbose=verbose,
        )

        return self.history

    def predict(self, X: np.ndarray, batch_size: int | None = None) -> np.ndarray:
        """预测"""
        n_gpus = self.strategy.num_replicas_in_sync
        if batch_size is None:
            batch_size = 128 * n_gpus
        return self.model.predict(X, batch_size=batch_size, verbose=0)

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray, target_scaler=None) -> dict:
        """评估模型"""
        pred = self.predict(X_test)

        results = {"normalized": {}, "original": {}}

        logger.info("\n各步预测评估 (标准化数据):")
        logger.info("-" * 60)
        logger.info(f"{'Step':<8} {'MAE(负压)':<12} {'MAE(含氧量)':<12} {'RMSE':<12}")
        logger.info("-" * 60)

        for s in range(1, self.output_steps + 1):
            pred_s = pred[:, s - 1, :]
            true_s = y_test[:, s - 1, :]

            mae_per_target = np.mean(np.abs(true_s - pred_s), axis=0)
            rmse = np.sqrt(np.mean((true_s - pred_s) ** 2))

            results["normalized"][str(s)] = {
                "mae": float(np.mean(mae_per_target)),
                "rmse": float(rmse),
                "mae_pressure": float(mae_per_target[0]),
                "mae_oxygen": float(mae_per_target[1]),
            }
            logger.info(f"{s:<8} {mae_per_target[0]:<12.4f} {mae_per_target[1]:<12.4f} {rmse:<12.4f}")

        logger.info("-" * 60)

        # 反标准化评估
        if target_scaler is not None:
            n_samples = pred.shape[0]
            pred_flat = pred.reshape(-1, 2)
            true_flat = y_test.reshape(-1, 2)

            pred_original = target_scaler.inverse_transform(pred_flat)
            true_original = target_scaler.inverse_transform(true_flat)

            pred_original = pred_original.reshape(n_samples, self.output_steps, 2)
            true_original = true_original.reshape(n_samples, self.output_steps, 2)

            logger.info("反标准化后的评估指标:")
            logger.info("-" * 70)
            logger.info(f"{'Step':<8} {'MAE负压(Pa)':<16} {'MAE含氧量(%)':<16} {'RMSE':<12}")
            logger.info("-" * 70)

            for s in range(1, self.output_steps + 1):
                pred_s = pred_original[:, s - 1, :]
                true_s = true_original[:, s - 1, :]

                mae_per_target = np.mean(np.abs(true_s - pred_s), axis=0)
                rmse = np.sqrt(np.mean((true_s - pred_s) ** 2))

                results["original"][str(s)] = {
                    "mae": float(np.mean(mae_per_target)),
                    "rmse": float(rmse),
                    "mae_pressure": float(mae_per_target[0]),
                    "mae_oxygen": float(mae_per_target[1]),
                }
                logger.info(f"{s:<8} {mae_per_target[0]:<16.4f} {mae_per_target[1]:<16.4f} {rmse:<12.4f}")

            logger.info("-" * 70)

            total_pred = pred_original.reshape(-1, 2)
            total_true = true_original.reshape(-1, 2)
            results["summary"] = {
                "mae_pressure_mean": float(np.mean(np.abs(total_true[:, 0] - total_pred[:, 0]))),
                "mae_oxygen_mean": float(np.mean(np.abs(total_true[:, 1] - total_pred[:, 1]))),
                "rmse_mean": float(np.sqrt(np.mean((total_true - total_pred) ** 2))),
            }

        return results

    def summary(self) -> None:
        self.model.summary()

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.model.save(path / "model.keras")
        with open(path / "config.json", "w") as f:
            json.dump({
                "seq_length": self.seq_length,
                "n_features": self.n_features,
                "output_steps": self.output_steps,
                "hidden_units": self.hidden_units,
                "dropout_rate": self.dropout_rate,
                "learning_rate": self.learning_rate,
                "l2_reg": self.l2_reg,
                "smoothness_weight": self.smoothness_weight,
            }, f, indent=2)
        logger.info(f"模型已保存: {path}")

    @classmethod
    def load(cls, path: str | Path, strategy: tf.distribute.Strategy | None = None) -> "LSTM":
        path = Path(path)
        with open(path / "config.json", "r") as f:
            config = json.load(f)
        instance = cls(**config, strategy=strategy)
        with instance.strategy.scope():
            instance.model = keras.models.load_model(
                path / "model.keras",
                custom_objects={
                    "PhysicsCouplingLayer": PhysicsCouplingLayer,
                    "PredictStepEmbedding": PredictStepEmbedding,
                    "physics_guided_loss": physics_guided_loss,
                },
            )
        return instance

__all__ = [
    "LSTM",
    "PhysicsCouplingLayer",
    "PredictStepEmbedding"
]