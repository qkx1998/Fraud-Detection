num = 10 # 设定分箱数
good = 0
bad = 1

# 排序，设定固定分箱数的分箱
df_ks = df.sort_values('pred').reset_index(drop=True)
df_ks['rank'] = np.floor((df_ks.index / len(df_ks) * num) + 1)
df_ks['set_1'] = 1

# 统计结果
res_ks = pd.DataFrame()
res_ks['grp_sum'] = df_ks.groupby('rank')['set_1'].sum()
res_ks['grp_min'] = df_ks.groupby('rank')['pred'].min()
res_ks['grp_max'] = df_ks.groupby('rank')['pred'].max()
res_ks['grp_mean'] = df_ks.groupby('rank')['pred'].mean()

# 最后一行添加汇总数据
res_ks.loc['total', 'grp_sum'] = df_ks['set_1'].sum()
res_ks.loc['total', 'grp_min'] = df_ks['pred'].min()
res_ks.loc['total', 'grp_max'] = df_ks['pred'].max()
res_ks.loc['total', 'grp_mean'] = df_ks['pred'].mean()

# 好用户的统计
res_ks['good_sum'] = df_ks[df_ks['label'] == good].groupby('rank')['set_1'].sum()
res_ks.good_sum.replace(np.nan, 0, inplace=True)
res_ks.loc['total', 'good_sum'] = res_ks['good_sum'].sum()
res_ks['good_percent'] = res_ks['good_sum'] / res_ks.loc['total', 'good_sum']
res_ks['good_percent_cum'] = res_ks['good_sum'].cumsum() / res_ks.loc['total', 'good_sum']

# 坏用户的统计
res_ks['bad_sum'] = df_ks[df_ks['label'] == bad].groupby('rank')['set_1'].sum()
res_ks.bad_sum.replace(np.nan, 0, inplace=True)
res_ks.loc['total', 'bad_sum'] = res_ks['bad_sum'].sum()
res_ks['bad_percent'] = res_ks['bad_sum'] / res_ks.loc['total', 'bad_sum']
res_ks['bad_percent_cum'] = res_ks['bad_sum'].cumsum() / res_ks.loc['total', 'bad_sum']

# KS计算公式
res_ks['diff'] = np.abs(res_ks['bad_percent_cum'] - res_ks['good_percent_cum'])

# 得到最终结果
res_ks.loc['total', 'bad_percent_cum'] = np.nan
res_ks.loc['total', 'good_percent_cum'] = np.nan
res_ks.loc['total', 'diff'] = res_ks['diff'].max()
res_ks = res_ks.reset_index()


'''
其余的调包的方法：
'''
def cal_ks_by_scipy(proba_arr, target_arr):
  '''
    param proba_arr:  numpy array of shape (1,), 预测为1的概率.
    param target_arr: numpy array of shape (1,), 取值为0或1.
    示例：
    >>> ks_compute(proba_arr=df['score'], target_arr=df[target])
    >>> 0.5262199213881699
    '''
    from scipy.stats import ks_2samp
    get_ks = lambda proba_arr, target_arr: ks_2samp(proba_arr[target_arr == 1], \
                                           proba_arr[target_arr == 0]).statistic
    ks_value = get_ks(proba_arr, target_arr)
    return ks_value
  
  
def cal_ks_by_cross(data,pred,y_label):
     '''
    功能: 计算KS值，输出对应分割点和累计分布函数曲线图
    'ks': KS值
    'crossdens': 好坏客户累积概率分布以及其差值gap
    '''
    crossfreq = pd.crosstab(data[pred],data[y_label])
    crossdens = crossfreq.cumsum(axis=0) / crossfreq.sum()
    crossdens['gap'] = abs(crossdens[0] - crossdens[1])
    ks = crossdens[crossdens['gap'] == crossdens['gap'].max()]
    return ks,crossdens
  







