from collections.abc import Mapping

from torch import Tensor, nn

from marine.models.embedding import SimpleEmbedding


class BaseModel(nn.Module):
    def __init__(
        self,
        embedding: SimpleEmbedding,
        encoders: Mapping[str, nn.Module],
        decoders: Mapping[str, nn.Module],
    ) -> None:
        super().__init__()
        self.embedding = embedding
        self.encoders = nn.ModuleDict(encoders)
        self.decoders = nn.ModuleDict(decoders)

    def forward(
        self,
        task: str,
        embedding_features: dict[str, Tensor],
        lengths: Tensor,
        mask: Tensor,
        prev_decoder_outputs: dict[str, Tensor] | None = None,
        decoder_targets: dict[str, Tensor] | None = None,
    ) -> Tensor | tuple[Tensor, ...]:
        embeddings = self.embedding(**embedding_features)
        encoder_outputs = self.encoders[task](embeddings, lengths)
        decoder_outputs = self.decoders[task](
            encoder_outputs, mask, prev_decoder_outputs, decoder_targets
        )

        return decoder_outputs
