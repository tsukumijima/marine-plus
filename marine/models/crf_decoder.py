from typing import Mapping, Sequence, cast

from marine.modules.crf_tagger import ConditionalRandomField
from torch import BoolTensor, Tensor, cat, nn


def _broadcast_tags(
    predicted_tags: Sequence[Sequence[int]], classfied: Tensor
) -> Tensor:
    class_probabilities = classfied * 0.0

    for i, instance_tags in enumerate(predicted_tags):
        for j, tag_id in enumerate(instance_tags):
            class_probabilities[i, j, tag_id] = 1

    return class_probabilities


class CRFDecoder(nn.Module):
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
        self.linear = nn.Linear(input_size, output_size)
        self.crf = ConditionalRandomField(output_size)

    def forward(
        self,
        logits: Tensor,
        mask: Tensor,
        prev_decoder_outputs: dict[str, Tensor] | None = None,
        decoder_targets: dict[str, Tensor] | None = None,
    ) -> tuple[Tensor, Tensor]:
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

        # CRFs
        best_paths = self.crf.viterbi_tags(linear_logits, cast(BoolTensor, mask))
        crf_logits = [x for x, _ in best_paths]
        crf_logits = _broadcast_tags(cast(list[list[int]], crf_logits), linear_logits)

        return linear_logits, crf_logits
