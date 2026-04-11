"""Two-Tower model definition for candidate retrieval."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

try:
    import torch
    from torch import Tensor, nn
    from torch.nn import functional as F
except ImportError:  # pragma: no cover - keep imports lightweight until training env is ready
    torch = None
    Tensor = Any  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]


if nn is None:
    class _BaseModule:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            _require_torch()
else:
    _BaseModule = nn.Module


def _require_torch() -> None:
    if torch is None or nn is None or F is None:
        raise ImportError("torch is required to use the Two-Tower model. Install torch before training or inference.")


@dataclass(slots=True)
class TowerConfig:
    """Configuration for one side of the Two-Tower model."""

    categorical_cardinalities: list[int]
    numeric_dim: int
    embedding_dim: int = 32
    hidden_dims: tuple[int, ...] = (128, 64)
    dropout: float = 0.1


@dataclass(slots=True)
class TwoTowerConfig:
    """Top-level configuration for the retrieval model."""

    user_tower: TowerConfig
    item_tower: TowerConfig
    output_dim: int = 64
    l2_normalize: bool = True


class FeatureTower(_BaseModule):
    """Encode categorical and numeric features into one dense embedding."""

    def __init__(self, config: TowerConfig) -> None:
        _require_torch()
        super().__init__()
        self.config = config

        self.categorical_embeddings = nn.ModuleList(
            nn.Embedding(num_embeddings=max(cardinality, 2), embedding_dim=config.embedding_dim)
            for cardinality in config.categorical_cardinalities
        )

        categorical_input_dim = len(config.categorical_cardinalities) * config.embedding_dim
        numeric_input_dim = config.numeric_dim
        input_dim = categorical_input_dim + numeric_input_dim

        if input_dim <= 0:
            raise ValueError("FeatureTower requires at least one categorical or numeric feature.")

        layers: list[nn.Module] = []
        previous_dim = input_dim
        for hidden_dim in config.hidden_dims:
            layers.append(nn.Linear(previous_dim, hidden_dim))
            layers.append(nn.ReLU())
            if config.dropout > 0:
                layers.append(nn.Dropout(config.dropout))
            previous_dim = hidden_dim
        self.mlp = nn.Sequential(*layers) if layers else nn.Identity()
        self.output_dim = previous_dim

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize embeddings and linear layers with stable defaults."""

        for embedding in self.categorical_embeddings:
            nn.init.xavier_uniform_(embedding.weight)
        for module in self.mlp.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, categorical: Tensor, numeric: Tensor | None = None) -> Tensor:
        if categorical.ndim != 2:
            raise ValueError(f"categorical features must be rank-2 [batch, fields], got shape={tuple(categorical.shape)}")

        categorical_parts: list[Tensor] = []
        for index, embedding in enumerate(self.categorical_embeddings):
            categorical_parts.append(embedding(categorical[:, index]))
        if categorical_parts:
            categorical_tensor = torch.cat(categorical_parts, dim=-1)
        else:
            categorical_tensor = torch.empty((categorical.shape[0], 0), device=categorical.device, dtype=torch.float32)

        if numeric is None:
            numeric_tensor = torch.empty((categorical.shape[0], 0), device=categorical.device, dtype=torch.float32)
        else:
            if numeric.ndim != 2:
                raise ValueError(f"numeric features must be rank-2 [batch, fields], got shape={tuple(numeric.shape)}")
            numeric_tensor = numeric.float()

        combined = torch.cat([categorical_tensor, numeric_tensor], dim=-1)
        return self.mlp(combined)


class TwoTowerModel(_BaseModule):
    """User tower + item tower retrieval model with dot-product scoring."""

    def __init__(self, config: TwoTowerConfig) -> None:
        _require_torch()
        super().__init__()
        self.config = config

        self.user_tower = FeatureTower(config.user_tower)
        self.item_tower = FeatureTower(config.item_tower)
        self.user_projection = nn.Linear(self.user_tower.output_dim, config.output_dim)
        self.item_projection = nn.Linear(self.item_tower.output_dim, config.output_dim)

        nn.init.xavier_uniform_(self.user_projection.weight)
        nn.init.zeros_(self.user_projection.bias)
        nn.init.xavier_uniform_(self.item_projection.weight)
        nn.init.zeros_(self.item_projection.bias)

    def encode_user(self, user_categorical: Tensor, user_numeric: Tensor | None = None) -> Tensor:
        """Project user-side features into the retrieval embedding space."""

        user_hidden = self.user_tower(user_categorical, user_numeric)
        user_embedding = self.user_projection(user_hidden)
        if self.config.l2_normalize:
            user_embedding = F.normalize(user_embedding, dim=-1)
        return user_embedding

    def encode_item(self, item_categorical: Tensor, item_numeric: Tensor | None = None) -> Tensor:
        """Project item-side features into the retrieval embedding space."""

        item_hidden = self.item_tower(item_categorical, item_numeric)
        item_embedding = self.item_projection(item_hidden)
        if self.config.l2_normalize:
            item_embedding = F.normalize(item_embedding, dim=-1)
        return item_embedding

    def forward(
        self,
        user_categorical: Tensor,
        item_categorical: Tensor,
        user_numeric: Tensor | None = None,
        item_numeric: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Encode one batch of positive pairs and return similarity outputs."""

        user_embedding = self.encode_user(user_categorical=user_categorical, user_numeric=user_numeric)
        item_embedding = self.encode_item(item_categorical=item_categorical, item_numeric=item_numeric)

        logits = torch.matmul(user_embedding, item_embedding.transpose(0, 1))
        positive_scores = (user_embedding * item_embedding).sum(dim=-1)
        return {
            "user_embedding": user_embedding,
            "item_embedding": item_embedding,
            "logits": logits,
            "positive_scores": positive_scores,
        }


def build_two_tower_config_from_metadata(metadata: dict[str, Any]) -> TwoTowerConfig:
    """Construct model configuration from dataset encoder metadata."""

    encoder_metadata = metadata.get("encoder", {})
    user_vocabularies = encoder_metadata.get("user_vocabularies", {})
    item_vocabularies = encoder_metadata.get("item_vocabularies", {})
    schema = encoder_metadata.get("schema", {})

    user_cardinalities = [
        len(user_vocabularies[column]["index_to_token"])
        for column in schema.get("user_categorical_columns", [])
    ]
    item_cardinalities = [
        len(item_vocabularies[column]["index_to_token"])
        for column in schema.get("item_categorical_columns", [])
    ]

    return TwoTowerConfig(
        user_tower=TowerConfig(
            categorical_cardinalities=user_cardinalities,
            numeric_dim=len(schema.get("user_numeric_columns", [])),
        ),
        item_tower=TowerConfig(
            categorical_cardinalities=item_cardinalities,
            numeric_dim=len(schema.get("item_numeric_columns", [])),
        ),
    )
