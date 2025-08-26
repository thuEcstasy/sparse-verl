import pandas as pd

# 使用 head 方法代替 nrows
df = pd.read_parquet('/home/haizhonz/data/gsm8k/test.parquet').head(128)
df.to_parquet('/home/haizhonz/data/gsm8k/test_128.parquet', index=False)
print(f"成功提取并保存了 {len(df)} 行数据")