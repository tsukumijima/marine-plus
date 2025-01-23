from typing import Mapping

from marine.data.feature.feature_set import FeatureSet
from torch import Tensor, cat, nn


class SimpleEmbedding(nn.Module):
    def __init__(
        self,
        embeding_sizes: Mapping[str, int],
        dropout: float | None,
        feature_set: FeatureSet,
    ) -> None:
        super().__init__()

        # embeddings
        self.embeddings = nn.ModuleDict(
            {
                key: nn.Embedding(
                    len(feature_set.feature_to_id[key]),
                    embeding_sizes[key],
                    padding_idx=feature_set.feature_to_id[key][feature_set.pad_token],
                )
                for key in feature_set.feature_keys
            }
        )

        if dropout:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

    def forward(self, **kwargs: dict[str, Tensor]) -> Tensor:
        # Embedding -> B * T * Embedding-size
        embs = [self.embeddings[key](kwargs[key]) for key in self.embeddings.keys()]
        embs = cat(embs, dim=2)

        if self.dropout:
            embs = self.dropout(embs)

        return embs
