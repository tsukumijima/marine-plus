from logging import getLogger
from pathlib import Path
from typing import Any, cast

import joblib
import numpy as np
from marine.data.feature.feature_table import (
    FEATURE_TABLES,
    PUNCTUATIONS,
    parse_accent_con_type,
)
from marine.types import BatchFeature, MarineFeature
from marine.utils.g2p_util import pron2mora
from numpy.typing import NDArray

logger = getLogger(__name__)


class FeatureSet(object):
    """
    A converter for embedding features
    """

    def __init__(
        self,
        vocab_path: str | Path,
        feature_table_key: str = "unidic-csj",
        feature_keys: list[str] | None = None,
        pad_token: str = "[PAD]",
        unk_token: str = "[UNK]",
    ) -> None:
        self.vocab_path = Path(vocab_path)

        self.pad_token = pad_token
        self.unk_token = unk_token
        self.default_tokens = [self.pad_token, self.unk_token]
        self.feature_table = FEATURE_TABLES[feature_table_key]

        if feature_keys:
            self.feature_keys = feature_keys
        else:
            self.feature_keys = list(self.feature_table.keys())

        self.feature_to_id = {key: {} for key in self.feature_keys}
        self.id_to_feature = {key: {} for key in self.feature_keys}

        self.init_feature_set()

    def _load_vocab(self) -> list[str]:
        if not self.vocab_path.exists():
            logger.error(f"Vocab has not found : {self.vocab_path}")
            raise FileNotFoundError(f"Vocab has not found : {self.vocab_path}")

        vocab = joblib.load(self.vocab_path)
        logger.info(f"Vocab loaded from {self.vocab_path} : {len(vocab)} words")

        return vocab

    def init_feature_set(self) -> None:
        self._load_vocab()

        for key in self.feature_keys:
            if key == "surface":
                feature_set = self.default_tokens + self._load_vocab()
            else:
                if key not in self.feature_table.keys():
                    raise ValueError(
                        f"Feature key must be one of {self.feature_table.keys()}"
                    )
                feature_set = self.default_tokens + (self.feature_table[key] or [])

            feature_to_id = {
                feature_value: index for index, feature_value in enumerate(feature_set)
            }
            id_to_feature = {
                index: feature_value for feature_value, index in feature_to_id.items()
            }
            self.feature_to_id[key] = feature_to_id
            self.id_to_feature[key] = id_to_feature

    def convert_feature_to_id(
        self, feature_key: str, features: list[str | int]
    ) -> NDArray[np.uint8]:
        if feature_key not in self.feature_to_id:
            raise ValueError(
                f"Not initialized feature key: the key must be one of {self.feature_to_id}"
            )

        return np.array(
            [
                self.feature_to_id[feature_key].get(
                    value, self.feature_to_id[feature_key][self.unk_token]
                )
                for value in features
            ],
            dtype=np.uint8,
        )

    def convert_id_to_feature(self, feature_key: str, ids: list[int]) -> NDArray[Any]:
        if feature_key not in self.id_to_feature:
            raise ValueError(
                f"Not initialized feature key: the key must be one of {self.id_to_feature}"
            )

        return np.array(
            [
                self.id_to_feature[feature_key].get(value, self.unk_token)
                for value in ids
            ]
        )

    def convert_nodes_to_feature(self, nodes: list[MarineFeature]) -> BatchFeature:
        """
        Input: dict型のリスト
        example:
        [
          {
            "surface": "今回",
            "pron": "コンカイ",
            "pos": "名詞:副詞可能:*:*",
            "c_type": "*",
            "c_form": "*",
            "accent_type": 1,
            "accent_con_type": "C1",
            "chain_flag": -1
          },
          ...
        ]
        """

        features = {key: np.array([], dtype=np.uint8) for key in self.feature_to_id}

        # init morph boundary for inference
        features["morph_boundary"] = np.array([], dtype=np.uint8)

        for node in nodes:
            mora = self.convert_feature_to_id(
                "mora",
                cast(
                    list[str | int],
                    pron2mora(node["pron"]) if node["pron"] else [node["surface"]],
                ),
            )

            morph_boundary = np.array([1] + ([0] * (len(mora) - 1)), dtype=np.uint8)

            # Push features
            features["mora"] = np.concatenate([features["mora"], mora], axis=0)
            features["morph_boundary"] = np.concatenate(
                [features["morph_boundary"], morph_boundary], axis=0
            )

            for key, table in self.feature_to_id.items():
                if key in ["mora", "morph_boundary"]:
                    continue

                if key == "a_con_type":
                    key = parse_accent_con_type(
                        node["accent_con_type"], node["pos"], unk_token=self.unk_token
                    )

                feature = table.get(node[key], table[self.unk_token])
                feature = np.array([feature] * len(mora), dtype=np.uint8)
                features[key] = np.concatenate([features[key], feature], axis=0)

        # First Mora could not be boundary
        # (boundary should be [0, 0, 1, 0, 0 ...])
        features["morph_boundary"][0] = 0

        return cast(BatchFeature, features)

    def get_punctuation_ids(self) -> list[int]:
        return [self.feature_to_id["mora"][punctuation] for punctuation in PUNCTUATIONS]
