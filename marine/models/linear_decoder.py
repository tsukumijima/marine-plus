from collections.abc import Mapping, Sequence

from torch import Tensor, cat, nn


class LinearDecoder(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        prev_task_embedding_label_list: Sequence[str] | None = None,
        prev_task_embedding_label_size: Mapping[str, int] | None = None,
        prev_task_embedding_size: Mapping[str, int] | None = None,
        prev_task_dropout: float | None = None,
        padding_idx: int = 0,
    ) -> None:
        super().__init__()
        if (
            prev_task_embedding_label_size
            and prev_task_embedding_label_list
            and prev_task_embedding_size
        ):
            embeddings = {}
            dropouts = {}

            for key in prev_task_embedding_label_list:
                embeddings[key] = nn.Embedding(
                    prev_task_embedding_label_size[key],
                    prev_task_embedding_size[key],
                    padding_idx=padding_idx,
                )
                input_size += prev_task_embedding_size[key]

                if prev_task_dropout:
                    dropouts[key] = nn.Dropout(prev_task_dropout)

            self.prev_task_embedding = nn.ModuleDict(embeddings)

            if len(dropouts) > 0:
                self.prev_task_dropout = nn.ModuleDict(dropouts)
            else:
                self.prev_task_dropout = None

        else:
            self.prev_task_embedding = None
            self.prev_task_dropout = None

        # NOTE: output_size must includes size for [PAD]
        self.linear = nn.Linear(input_size, output_size, bias=True)

    def forward(
        self,
        logits: Tensor,
        mask: Tensor,
        prev_decoder_outputs: dict[str, Tensor] | None = None,
        decoder_targets: Tensor | None = None,
    ) -> Tensor:
        if self.prev_task_embedding is not None and prev_decoder_outputs is not None:
            prev_decoder_output_embs = []

            for key in self.prev_task_embedding.keys():
                prev_decoder_output = prev_decoder_outputs[key]
                prev_decoder_output_emb = self.prev_task_embedding[key](
                    prev_decoder_output
                )

                if self.prev_task_dropout:
                    prev_decoder_output_emb = self.prev_task_dropout[key](
                        prev_decoder_output_emb
                    )

                prev_decoder_output_embs.append(prev_decoder_output_emb)

            logits = cat([logits] + prev_decoder_output_embs, dim=2)

        # Linear -> B * T * Output-size
        linear_logits = self.linear(logits)

        return linear_logits
