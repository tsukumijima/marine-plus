import os
import re
from pathlib import Path

import jaconv
import pyopenjtalk

EDIT_DICT_CSV = str(Path(__file__).parent.joinpath("edit_dict.csv"))
USER_DICT_CSV = str(Path(__file__).parent.joinpath("user_dict.csv"))
USER_DICT_DIC = str(Path(__file__).parent.joinpath("user_dict.dic"))


def load_user_dict():
    if os.path.isfile(USER_DICT_CSV):
        if sum([1 for _ in open(USER_DICT_CSV, encoding="utf-8")]) != 0:
            pyopenjtalk.create_user_dict(USER_DICT_CSV, USER_DICT_DIC)
            print(USER_DICT_DIC)
            pyopenjtalk.set_user_dict(USER_DICT_DIC)


def dictupdate_byrecord(record, dict_records=None):
    SMALL_HIRA = "ぁぃぅぇぉゃゅょ"
    # CSVを入れてほしいけどtsvでもOK
    word, yomi_acc, c1c2 = re.split("[,\t]", record)

    yomi_acc = yomi_acc.replace("[", "↗").replace("]", "↘")
    yomi_acc = jaconv.kata2hira(yomi_acc)  # 間違ってカタカナで入れていた場合
    C1C2 = c1c2.upper()  # 間違って小文字で入れていた場合
    p = re.compile(
        "[\u3041-\u309F]*↗[\u3041-\u309Fー]+↘[\u3041-\u309Fー]*"
    )  # ひらがなと1回の[]だけで構成されている
    # 記載ミスがあれば処理しない
    if not (p.fullmatch(yomi_acc) and C1C2 in ("C1", "C2")):
        print(
            '{0}は"読み"又は"C1C2"の書き方に誤りがあったため登録をスキップしました。'.format(
                word
            )
        )
        return dict_records

    yomi = jaconv.hira2kata(re.sub(r"[↗↘]", "", yomi_acc))
    accent_type = (
        "0"
        if c1c2.upper() == "C2"
        else str(re.sub(r"[%s]" % SMALL_HIRA, "", yomi_acc).find("↘") - 1)
    )
    dict_record = "{0},,,1,名詞,一般,*,*,*,*,{0},{1},{1},{2},{3}\n".format(
        word, yomi, "{0}/{1}".format(accent_type, len(yomi)), C1C2
    )

    if dict_records is not None:
        dict_records += dict_record
        return dict_records
    else:
        with open(USER_DICT_CSV, "a", encoding="utf-8") as f:
            f.write(dict_record)
        load_user_dict()


def make_dict(csv_dir, is_rebuild=True):
    input_file = csv_dir
    with open(input_file, encoding="utf-8") as f:
        lines = f.readlines()

    dict_records = ""
    for line in lines:
        line = line.strip().replace(" ", "")
        dict_records = dictupdate_byrecord(line, dict_records)

    if is_rebuild:
        os.remove(USER_DICT_CSV)
    with open(USER_DICT_CSV, "a", encoding="utf-8") as f:
        f.write(dict_records)
    load_user_dict()


make_dict(EDIT_DICT_CSV)
