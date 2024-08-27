import difflib
import re
import warnings

import numpy as np
import pykakasi
from marine.data.feature.feature_table import RAW_FEATURE_KEYS

kakasi = pykakasi.kakasi()
BOIN_DICT = {"a": "ア", "i": "イ", "u": "ウ", "e": "エ", "o": "オ", "n": "ン"}


OPEN_JTALK_FEATURE_INDEX_TABLE = {
    "surface": 0,
    "pos": [1, 2, 3, 4],
    "c_type": 5,
    "c_form": 6,
    "pron": 9,
    "accent_type": 10,
    "accent_con_type": 11,
    "chain_flag": 12,
}
OPEN_JTALK_FEATURE_RENAME_TABLE = {
    "surface": "string",
    "pos": ["pos", "pos_group1", "pos_group2", "pos_group3"],
    "c_type": "ctype",
    "c_form": "cform",
    "accent_type": "acc",
    "pron": "pron",
    "accent_con_type": "chain_rule",
    "chain_flag": "chain_flag",
}

PUNCTUATION_FULL_TO_HALF_TABLE = {
    "、": ",",
    "。": ".",
    "？": "?",
    "！": "!",
}
PUNCTUATION_FULL_TO_HALF_TRANS = str.maketrans(PUNCTUATION_FULL_TO_HALF_TABLE)


def convert_open_jtalk_node_to_feature(nodes):
    features = []
    raw_feature_keys = RAW_FEATURE_KEYS["open-jtalk"]
    pre_pron = None

    for node in nodes:
        # parse feature
        node_feature = {}
        for feature_key in raw_feature_keys:
            jtalk_key = OPEN_JTALK_FEATURE_RENAME_TABLE[feature_key]

            if feature_key == "pos":
                value = ":".join([node[_k] for _k in jtalk_key])
            elif feature_key == "accent_type":
                value = int(node[jtalk_key])
            elif feature_key == "accent_con_type":
                value = node[jtalk_key].replace("/", ",")
            elif feature_key == "chain_flag":
                value = int(node[jtalk_key])
            elif feature_key == "pron":
                if node[jtalk_key][0] == "ー":
                    try:
                        value = trans_hyphen2katakana(pre_pron + node[jtalk_key])[
                            -len(node[jtalk_key]) :
                        ]
                    except Exception:
                        print(node[jtalk_key])
                        value = node[jtalk_key]
                    pre_pron = value
                else:
                    value = node[jtalk_key].replace("’", "").replace("ヲ", "オ")
                    try:
                        value = trans_hyphen2katakana(value)
                    except Exception:
                        print(value)

                    pre_pron = value
            else:
                value = node[jtalk_key]

            node_feature[feature_key] = value

        if node_feature["surface"] == "・":
            continue
        elif node_feature["surface"] in PUNCTUATION_FULL_TO_HALF_TABLE.keys():
            surface = node_feature["surface"].translate(PUNCTUATION_FULL_TO_HALF_TRANS)
            pron = None
            node_feature["surface"] = surface
            node_feature["pron"] = pron

        features.append(node_feature)

    return features


def convert_njd_feature_to_marine_feature(njd_features):
    marine_features = []

    raw_feature_keys = RAW_FEATURE_KEYS["open-jtalk"]
    for njd_feature in njd_features:
        marine_feature = {}
        for feature_key in raw_feature_keys:
            if feature_key == "pos":
                value = ":".join(
                    [
                        njd_feature["pos"],
                        njd_feature["pos_group1"],
                        njd_feature["pos_group2"],
                        njd_feature["pos_group3"],
                    ]
                )
            elif feature_key == "accent_con_type":
                value = njd_feature["chain_rule"].replace("/", ",")
            elif feature_key == "pron":
                value = njd_feature["pron"].replace("’", "").replace("ヲ", "オ")
            else:
                value = njd_feature[OPEN_JTALK_FEATURE_RENAME_TABLE[feature_key]]
            marine_feature[feature_key] = value

        if marine_feature["surface"] == "・":
            continue
        elif marine_feature["surface"] in PUNCTUATION_FULL_TO_HALF_TABLE.keys():
            surface = marine_feature["surface"].translate(
                PUNCTUATION_FULL_TO_HALF_TRANS
            )
            pron = None
            marine_feature["surface"] = surface
            marine_feature["pron"] = pron

        marine_features.append(marine_feature)

    return marine_features


def convert_open_jtalk_format_label(
    labels,
    morph_boundaries,
    accent_nucleus_label=1,
    accent_phrase_boundary_label=1,
    morph_boundary_label=1,
):
    assert "accent_status" in labels.keys(), "`accent_status` is missing in labels"
    assert (
        "accent_phrase_boundary" in labels.keys()
    ), "`accent_phrase_boundary` is missing in labels"

    # squeeze results
    mora_accent_status = labels["accent_status"][0]
    mora_accent_phrase_boundary = labels["accent_phrase_boundary"][0]
    morph_boundary = morph_boundaries[0]

    assert len(mora_accent_status) == len(mora_accent_phrase_boundary), (
        "Not match sequence lenght between"
        "`accent_status`, `morph_boundary`, and `accent_phrase_boundary`"
    )

    mora_accent_phrase_boundary = np.array(mora_accent_phrase_boundary)

    # convert mora-based accent phrase boundary label to morph-based label
    morph_boundary_indexes = np.where(morph_boundary == morph_boundary_label)[0]
    morph_accent_phrase_boundary = np.split(
        mora_accent_phrase_boundary, morph_boundary_indexes
    )
    # `chain_flag` in OpenJTalk represents the status whether the morph will be connected
    morph_accent_phrase_boundary = [
        0 if boundary[0] == accent_phrase_boundary_label else 1
        for boundary in morph_accent_phrase_boundary
    ]
    # first `chain_flag` must be -1
    morph_accent_phrase_boundary[0] = -1
    num_boundary = morph_accent_phrase_boundary.count(0) + 1

    # convert mora-based accent status label to ap-based label
    # アクセント句境界かつ形態素句境界のindexを取得に修正
    mora_accent_phrase_boundary_indexes = np.where(
        mora_accent_phrase_boundary + morph_boundary
        == accent_phrase_boundary_label + morph_boundary_label
    )[0]
    phrase_accent_statuses = np.split(
        mora_accent_status, mora_accent_phrase_boundary_indexes
    )
    phrase_accent_status_labels = []

    for phrase_accent_status in phrase_accent_statuses:
        accent_nucleus_indexes = np.where(phrase_accent_status == accent_nucleus_label)[
            0
        ]
        if len(accent_nucleus_indexes) == 0:
            accent_nucleus_index = 0
        else:
            accent_nucleus_index = accent_nucleus_indexes[0] + 1
        phrase_accent_status_labels.append(accent_nucleus_index)

    if len(phrase_accent_status_labels) > num_boundary:
        warnings.warn(
            (
                "Lenght of AP-based accent status will be adjusted "
                "by morph-based accent phrase boundary: "
                f"{len(phrase_accent_status_labels)} > {num_boundary}"
            ),
            stacklevel=2,
        )
        phrase_accent_status_labels = phrase_accent_status_labels[:num_boundary]

    # convert mora-based accent status to morph-based label
    # the accent label for OpenJTalk pushed in first morph
    morph_accent_status = [
        phrase_accent_status_labels.pop(0) if morph_accent_phrase_flag < 1 else 0
        for morph_accent_phrase_flag in morph_accent_phrase_boundary
    ]

    return {
        "accent_status": morph_accent_status,
        "accent_phrase_boundary": morph_accent_phrase_boundary,
    }


def trans_hyphen2katakana(text):
    """
    伸ばし棒をカタカナに変換
    例：きょー→きょお
    """
    hyphen_string_list = re.findall("..ー", text)
    text = replace_hyphen(text, hyphen_string_list)

    hyphen_string_list = re.findall(".ー", text)
    text = replace_hyphen(text, hyphen_string_list)

    return text


def replace_hyphen(text, hyphen_string_list):
    for _str in hyphen_string_list:
        if "[" in _str or "]" in _str:
            result = kakasi.convert(_str.replace("[", "").replace("]", ""))[0]
        else:
            _str_wo_hyphen = _str.replace("ー", "")
            result = kakasi.convert(_str_wo_hyphen)[-1]

        transed_hyphen_string = _str[:-1] + BOIN_DICT[result["hepburn"][-1]]
        text = text.replace(_str, transed_hyphen_string)

    return text


def print_diff_hl(ground_truth, target):
    """
    文字列の差異をハイライト表示する
    """
    color_dic = {"red": "\033[31m", "green": "\033[32m", "end": "\033[0m"}

    d = difflib.Differ()
    diffs = d.compare(ground_truth, target)

    result = ""
    for diff in diffs:
        status, _, character = list(diff)
        if status == "-":
            character = color_dic["red"] + character + color_dic["end"]
        elif status == "+":
            character = color_dic["green"] + character + color_dic["end"]
        else:
            pass
        result += character

    print(f"ground truth : {ground_truth}")
    print(f"target string: {target}")
    print(f"diff result  : {result}")
