import pandas as pd
import numpy as np

# 给特征做分箱
bins = [0, 10, 20, 30, 40, 50, 100]
df[f'{col}_cate'] = pd.cut(df[col], bins, labels=["bin_1","bin_2","bin_3","bin_4","bin_5","bin_6"])

df['label_0'] = df['label'].map(lambda x: 1 if x == 0 else 0)
df['label_1'] = df['label'].map(lambda x: 1 if x == 1 else 0)

# 统计各个箱中好坏比率
woe_iv_df_1 = df.groupby('age_cate')['label_0'].agg({'cnt_0':'sum'}).reset_index()
woe_iv_df_2 = df.groupby('age_cate')['label_1'].agg({'cnt_1':'sum'}).reset_index()
woe_iv_df = pd.merge(woe_iv_df_1, woe_iv_df_2, how = 'left', on = f'{col}_cate')
woe_iv_df['cnt'] = woe_iv_df['cnt_0'] + woe_iv_df['cnt_1']

woe_iv_df['ratio_0'] = woe_iv_df['cnt_0']/woe_iv_df['cnt_0'].sum()
woe_iv_df['ratio_1'] = woe_iv_df['cnt_1']/woe_iv_df['cnt_1'].sum()

# 计算各个箱的woe值
woe_iv_df['woe'] = np.log(woe_iv_df['ratio_0']/woe_iv_df['ratio_1'])

# 由于是测试数据，就直接求iv值
woe_iv_df['iv'] = (woe_iv_df['ratio_0'] - woe_iv_df['ratio_1']) * woe_iv_df['woe']

iv = woe_iv_df['iv'].sum()
