from typing import Literal, TypedDict

import numpy as np
from numpy.typing import NDArray
from torch import Tensor


FeatureIDArray = NDArray[np.int64]
MorphBoundaryArray = NDArray[np.uint8]


# アクセント表現モード
AccentRepresentMode = Literal[
    "binary",  # アクセント核位置を1、それ以外を0で表現
    "high_low",  # 各モーラの高低を表現 (0=低 / 1=高)
]


class MarineFeature(TypedDict):
    """marine 内部で使用する形態素素性を表す型"""

    surface: str  # 表層形
    pron: str | None  # 発音形式
    pos: str  # 品詞 (例: "名詞:代名詞:一般:*")
    c_type: str  # 活用型
    c_form: str  # 活用形
    accent_type: int  # アクセント型
    accent_con_type: str  # アクセント結合型
    chain_flag: int  # アクセント句の連結フラグ


class OpenJTalkFormatLabel(TypedDict):
    """OpenJTalkフォーマットのラベルを表す型"""

    # fmt: off
    accent_status: list[int]  # アクセント核位置 (0: 無核, 1-n: n番目のモーラにアクセント核)
    accent_phrase_boundary: list[Literal[-1, 0, 1]]  # アクセント句境界 (-1: 文頭, 0: 非境界, 1: 境界)
    # fmt: on


class MarineLabel(TypedDict):
    """marine 内部で使用するラベルを表す型"""

    # fmt: off
    mora: list[list[str]]  # モーラ列 (例: [["コ", "ン", "ニ", "チ", "ワ"]])
    intonation_phrase_boundary: list[list[Literal[0, 1]]]  # イントネーション句境界 (0: 非境界, 1: 境界)
    accent_phrase_boundary: list[list[Literal[0, 1]]]  # アクセント句境界 (0: 非境界, 1: 境界)
    accent_status: list[list[Literal[0, 1]]]  # アクセント核位置 (binary: 0/1, high_low: 0/1)
    # fmt: on


class AnnotateLabel(TypedDict):
    """アノテーションラベルを表す型"""

    token_type: Literal["morph", "mora"]  # トークンの単位
    labels: list[list[int]] | list[Tensor]  # ラベル列


class PredictAnnotates(TypedDict, total=False):
    """推論時のアノテーションを表す型"""

    intonation_phrase_boundary: AnnotateLabel
    accent_phrase_boundary: AnnotateLabel
    accent_status: AnnotateLabel


class BatchFeature(TypedDict):
    """バッチの特徴量を表す型"""

    # 必須フィールド
    morph_boundary: MorphBoundaryArray  # 形態素境界情報
    # config.data.input_keys に依存するフィールド (推論に用いるモデルの config.yaml 定義次第では省略される)
    mora: FeatureIDArray  # モーラ ID 列
    surface: FeatureIDArray  # 表層形 ID 列
    pos: FeatureIDArray  # 品詞 ID 列
    c_type: FeatureIDArray  # 活用型 ID 列
    c_form: FeatureIDArray  # 活用形 ID 列
    accent_type: FeatureIDArray  # アクセント型 ID 列
    accent_con_type: FeatureIDArray  # アクセント結合型 ID 列
    chain_flag: FeatureIDArray  # アクセント句の連結フラグ ID 列 (現在の学習レシピでは未使用) # fmt: skip


class BatchItem(TypedDict):
    """バッチの各要素を表す型"""

    features: BatchFeature  # 特徴量
    labels: dict[str, list[int]] | None  # ラベル (推論時は None)
    ids: str | None  # スクリプトID (推論時は None)


class ModelInputs(TypedDict):
    """モデルの入力を表す型"""

    embedding_features: dict[str, Tensor]  # 埋め込み特徴量
    lengths: Tensor  # 系列長
    mask: Tensor  # マスク
    prev_decoder_outputs: dict[str, Tensor]  # 前のデコーダーの出力


class PadFeature(TypedDict):
    """パディングされた特徴量を表す型"""

    # 必須フィールド
    morph_boundary: list[list[list[int]]]  # 形態素境界情報
    # config.data.input_keys に依存するフィールド (推論に用いるモデルの config.yaml 定義次第では省略される)
    mora: Tensor  # モーラ ID 列
    mora_length: Tensor  # モーラ長 (config.data.input_length_key によって定義される)
    surface: Tensor  # 表層形 ID 列
    pos: Tensor  # 品詞 ID 列
    c_type: Tensor  # 活用型 ID 列
    c_form: Tensor  # 活用形 ID 列
    accent_type: Tensor  # アクセント型 ID 列
    accent_con_type: Tensor  # アクセント結合型 ID 列
    chain_flag: Tensor  # アクセント句の連結フラグ ID 列 (現在の学習レシピでは未使用)


class PadOutputLabel(TypedDict):
    """パディングされた出力ラベルを表す型"""

    label: Tensor  # ラベル列
    length: Tensor  # 系列長


class PadOutputs(TypedDict):
    """パディングされた出力を表す型"""

    intonation_phrase_boundary: PadOutputLabel
    accent_phrase_boundary: PadOutputLabel
    accent_status: PadOutputLabel
