from pathlib import Path

import pytest
import torch

from marine.predict import Predictor
from marine.types import MarineFeature


@pytest.fixture
def predictor() -> Predictor:
    """load inference model using default config"""
    return Predictor()


def test_predict(predictor: Predictor) -> None:
    """just to confirm predict() is working without errors."""
    nodes: list[MarineFeature] = [
        {
            "surface": "水",
            "pron": "ミズ",
            "pos": "名詞:一般:*:*",
            "c_type": "*",
            "c_form": "*",
            "accent_type": 0,
            "accent_con_type": "C3",
            "chain_flag": -1,
        },
        {
            "surface": "を",
            "pron": "オ",
            "pos": "助詞:格助詞:一般:*",
            "c_type": "*",
            "c_form": "*",
            "accent_type": 0,
            "accent_con_type": "動詞%F5,名詞%F1",
            "chain_flag": 1,
        },
        {
            "surface": "マレーシア",
            "pron": "マレーシア",
            "pos": "名詞:固有名詞:地域:国",
            "c_type": "*",
            "c_form": "*",
            "accent_type": 2,
            "accent_con_type": "C1",
            "chain_flag": 0,
        },
        {
            "surface": "から",
            "pron": "カラ",
            "pos": "助詞:格助詞:一般:*",
            "c_type": "*",
            "c_form": "*",
            "accent_type": 2,
            "accent_con_type": "名詞%F1",
            "chain_flag": 1,
        },
        {
            "surface": "買わ",
            "pron": "カワ",
            "pos": "動詞:自立:*:*",
            "c_type": "五段・ワ行促音便",
            "c_form": "未然形",
            "accent_type": 0,
            "accent_con_type": "*",
            "chain_flag": 0,
        },
        {
            "surface": "なく",
            "pron": "ナク",
            "pos": "助動詞:*:*:*",
            "c_type": "特殊・ナイ",
            "c_form": "連用テ接続",
            "accent_type": 1,
            "accent_con_type": "動詞%F3@0",
            "chain_flag": 1,
        },
        {
            "surface": "て",
            "pron": "テ",
            "pos": "助詞:接続助詞:*:*",
            "c_type": "*",
            "c_form": "*",
            "accent_type": 0,
            "accent_con_type": "動詞%F1,形容詞%F1,名詞%F5",
            "chain_flag": 1,
        },
        {
            "surface": "は",
            "pron": "ワ",
            "pos": "助詞:係助詞:*:*",
            "c_type": "*",
            "c_form": "*",
            "accent_type": 0,
            "accent_con_type": "名詞%F1,動詞%F2@0,形容詞%F2@0",
            "chain_flag": 1,
        },
        {
            "surface": "なら",
            "pron": "ナラ",
            "pos": "動詞:非自立:*:*",
            "c_type": "五段・ラ行",
            "c_form": "未然形",
            "accent_type": 2,
            "accent_con_type": "*",
            "chain_flag": 0,
        },
        {
            "surface": "ない",
            "pron": "ナイ",
            "pos": "助動詞:*:*:*",
            "c_type": "特殊・ナイ",
            "c_form": "基本形",
            "accent_type": 1,
            "accent_con_type": "動詞%F3@0,形容詞%F2@1",
            "chain_flag": 1,
        },
        {
            "surface": "の",
            "pron": "ノ",
            "pos": "名詞:非自立:一般:*",
            "c_type": "*",
            "c_form": "*",
            "accent_type": 2,
            "accent_con_type": "動詞%F2@0,形容詞%F2@-1",
            "chain_flag": 0,
        },
        {
            "surface": "です",
            "pron": "デス",
            "pos": "助動詞:*:*:*",
            "c_type": "特殊・デス",
            "c_form": "基本形",
            "accent_type": 1,
            "accent_con_type": "名詞%F2@1,動詞%F1,形容詞%F2@0",
            "chain_flag": 1,
        },
        {
            "surface": ".",
            "pron": None,
            "pos": "記号:句点:*:*",
            "c_type": "*",
            "c_form": "*",
            "accent_type": 0,
            "accent_con_type": "*",
            "chain_flag": 0,
        },
    ]

    print(predictor.predict([nodes], accent_represent_mode="binary"))
    print(predictor.predict([nodes], accent_represent_mode="high_low"))

    # If you want the format for OpenJTalk, `accent_represent_mode` will be fixed as `binary
    print(
        predictor.predict(
            [nodes],
            accent_represent_mode="binary",
            require_open_jtalk_format=True,
        )
    )


def test_predictor_accepts_path_arguments(predictor: Predictor) -> None:
    """Path 型の引数でも Predictor を初期化できることを確認する。"""

    path_predictor = Predictor(
        model_dir=Path(predictor.model_dir),
        postprocess_vocab_dir=Path(predictor.postprocess_vocab_dir),
    )

    assert path_predictor.model_dir == predictor.model_dir
    assert path_predictor.postprocess_vocab_dir == predictor.postprocess_vocab_dir


def test_pad_annotate_label_accepts_tensor_labels(predictor: Predictor) -> None:
    """Tensor のラベル列でも pad_annotate_label() が処理できることを確認する。"""

    result = predictor.pad_annotate_label(
        {
            "accent_phrase_boundary": {
                "token_type": "mora",
                "labels": [torch.tensor([0, 1, 0], dtype=torch.int64)],
            },
        },
        mora=[["ミ", "ズ", "ワ"]],
        morph_boundary=[],
    )

    assert torch.equal(
        result["accent_phrase_boundary"],
        torch.tensor([[0, 1, 0]], device=result["accent_phrase_boundary"].device),
    )


def test_predict_applies_postprocess_vocab(predictor: Predictor) -> None:
    """後処理辞書に載っている語 (音声合成) を含む入力でも推論でき、辞書の値が反映されることを確認する。"""

    nodes: list[MarineFeature] = [
        {
            "surface": "音声",
            "pron": "オンセー",
            "pos": "名詞:一般:*:*",
            "c_type": "*",
            "c_form": "*",
            "accent_type": 5,
            "accent_con_type": "C1",
            "chain_flag": -1,
        },
        {
            "surface": "合成",
            "pron": "ゴーセー",
            "pos": "名詞:サ変接続:*:*",
            "c_type": "*",
            "c_form": "*",
            "accent_type": 0,
            "accent_con_type": "C2",
            "chain_flag": 1,
        },
    ]

    result = predictor.predict([nodes])

    # 辞書 (marine/dict/accent_phrase_boundary/artifact.tsv) では「音声合成」は 1 つのアクセント句
    assert result["accent_phrase_boundary"] == [[0, 0, 0, 0, 0, 0, 0, 0]]
