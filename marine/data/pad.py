from typing import Any, cast

import numpy as np
import torch
from numpy.typing import NDArray
from torch.nn.utils.rnn import pad_sequence

from marine.types import BatchFeature, BatchItem, PadFeature, PadOutputs


class Padsequence:
    def __init__(
        self,
        input_keys: list[str],
        input_length_key: str,
        output_keys: list[str],
        num_classes: int,
        is_inference: bool = False,
        padding_idx: int = 0,
    ) -> None:
        self.input_keys = input_keys
        self.input_length_key = input_length_key
        self.output_keys = output_keys
        self.num_classes = num_classes
        self.is_inference = is_inference
        self.padding_idx = padding_idx

    def pad_feature(self, inputs: list[BatchFeature]) -> PadFeature:
        padded_feature: dict[str, Any] = {}

        for key in self.input_keys:
            feature = [
                torch.tensor(features[key], dtype=torch.int64) for features in inputs
            ]

            if key in self.input_length_key:
                padded_feature[f"{key}_length"] = torch.tensor(
                    [len(f) for f in feature], dtype=torch.int64
                )

            padded_x = pad_sequence(
                feature,
                batch_first=True,
                padding_value=self.padding_idx,
            )
            padded_feature[key] = padded_x

        return cast(PadFeature, padded_feature)

    def __call__(
        self, batch: list[BatchItem]
    ) -> tuple[
        PadFeature, PadOutputs | None, list[NDArray[np.uint8]], list[str] | None
    ]:
        # sort by length
        if not self.is_inference:
            batch = sorted(
                batch,
                key=lambda x: len(x["features"][self.input_length_key]),
                reverse=True,
            )

        inputs = [x["features"] for x in batch]
        padded_inputs = self.pad_feature(inputs)

        if not self.is_inference:
            padded_outputs = cast(
                PadOutputs,
                {
                    key: {
                        "label": pad_sequence(
                            [
                                # Convert 1-based label (for pad)
                                torch.tensor(x["labels"][key] + 1, dtype=torch.long)  # type: ignore
                                for x in batch
                            ],
                            batch_first=True,
                            padding_value=self.padding_idx,
                        ),
                        "length": torch.tensor([len(x["labels"][key]) for x in batch]),  # type: ignore
                    }
                    for key in self.output_keys
                },
            )
            script_ids = [x["ids"] for x in batch if x["ids"] is not None]  # type: ignore
        else:
            padded_outputs = None
            script_ids = None

        morph_boundary = [x["features"]["morph_boundary"] for x in batch]

        return padded_inputs, padded_outputs, morph_boundary, script_ids
