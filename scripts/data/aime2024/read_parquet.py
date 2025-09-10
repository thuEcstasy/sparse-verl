import pandas as pd

# 只读第一行
df = pd.read_parquet("test.parquet")
first_row = df.iloc[0]
print(first_row)
