'''Defines immutable hyperparameter configurations for all model experiments.'''

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Preprocessing:
    context_window: int = 4


@dataclass(frozen=True)
class MLP:
    hidden_layer_sizes: tuple = (256,)
    activation: str = "relu"
    solver: str = "adam"
    alpha: float = 0.0001
    batch_size: int = 200
    max_iter: int = 30
    verbose: bool = True


@dataclass(frozen=True)
class Hyperparameters:
    preprocessing: Preprocessing = field(
        default_factory=Preprocessing
    )
    
    mlp: MLP = field(default_factory=MLP)


# Shared configuration used by the whole program.
HYPERPARAMETERS = Hyperparameters()