import pandas as pd
from marine.utils.openjtalk_util import print_diff_hl

df = pd.read_csv(
    "../recipe/general/wrong_mora_info.csv",
    sep="|",
    names=["wav", "jtalk", "anotation"],
)

for i in df.iterrows():
    print(i[1]["wav"])
    print(print_diff_hl(i[1]["jtalk"], i[1]["anotation"]))
