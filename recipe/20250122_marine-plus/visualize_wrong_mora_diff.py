from pathlib import Path

import pandas as pd
import yaml

from marine.utils.openjtalk_util import print_diff_hl


# Load text.yaml
with open(Path(__file__).parent / "data" / "text.yaml", encoding="utf-8") as f:
    text_data = yaml.safe_load(f)

df = pd.read_csv(
    Path(__file__).parent / "wrong_mora_info.csv",
    sep="|",
    names=["wav", "jtalk", "anotation"],
)

for i in df.iterrows():
    wav_id = i[1]["wav"]
    print(f"\n========== {wav_id} ==========")
    if wav_id in text_data:
        print(f"       Text : {text_data[wav_id]['text_level0']}")
    print_diff_hl(i[1]["jtalk"], i[1]["anotation"])
