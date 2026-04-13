from src.modeling.lstm import (
    LSTM,
    physics_guided_loss,
    PhysicsCouplingLayer,
    PredictStepEmbedding,
)
from src.modeling.transformer import (
    PositionalEncoding, LearnablePositionalEncoding, MultiHeadSelfAttention,
    TransformerDecoderLayer, TransformerEncoderLayer, PhysicsCouplingLayer,
    MultiStepTransformer, combined_loss, setup_gpu
)
from src.modeling.optimization import (
    OptimizationConfig,
    OptimizationResult,
    MultiObjectiveResult,
    SceneDetector,
    BaseOptimizer,
    BayesianOptimizer,
    HierarchicalBayesianOptimizer,
    HybridTwoStageOptimizer,
    MultiObjectiveOptimizer,
    ControlRecommender,
    create_optimizer,
)
from src.modeling.narx_lstm import (
    NARXLSTM,
    NARXLSTMTrainer,
    create_narx_lstm_model,
)
from src.modeling.physics_loss import (
    PhysicsConstraintLoss,
    CombinedLoss,
    create_physics_loss,
)
from src.modeling.pinn_proxy import (
    PINNProxyMLP,
    ProxyTrainer,
    create_proxy_model,
)

__all__ = [
    # LSTM
    "LSTM",
    "physics_guided_loss",
    "PhysicsCouplingLayer",
    "PredictStepEmbedding",
    # Transformer
    "PositionalEncoding",
    "LearnablePositionalEncoding",
    "MultiHeadSelfAttention",
    "TransformerDecoderLayer",
    "TransformerEncoderLayer",
    "MultiStepTransformer",
    "combined_loss",
    "setup_gpu",
    # Optimization
    "OptimizationConfig",
    "OptimizationResult",
    "MultiObjectiveResult",
    "SceneDetector",
    "BaseOptimizer",
    "BayesianOptimizer",
    "HierarchicalBayesianOptimizer",
    "HybridTwoStageOptimizer",
    "MultiObjectiveOptimizer",
    "ControlRecommender",
    "create_optimizer",
    # NARX-LSTM (新增)
    "NARXLSTM",
    "NARXLSTMTrainer",
    "create_narx_lstm_model",
    # Physics Loss (新增)
    "PhysicsConstraintLoss",
    "CombinedLoss",
    "create_physics_loss",
    # PINN Proxy (新增)
    "PINNProxyMLP",
    "ProxyTrainer",
    "create_proxy_model",
]