# 获取某列特征， 转换为dataframe, 命名为score
base_df = pd.DataFrame(list(train[col]), columns=['score'])
test_df = pd.DataFrame(list(test[col]), columns=['score'])

# 统计非缺失值数量 和 缺失值数量
base_notnull_cnt = base_df['score'].dropna().shape[0]
test_notnull_cnt = test_df['score'].dropna().shape[0]

base_null_cnt = len(base_df) - base_notnull_cnt
test_null_cnt = len(test_df) - test_notnull_cnt

# 设置箱子数 和 箱子内最小样本数
bins = 20 
min_sample = 10 

# 得到分位数列表
q_list = []
bin_num = min(bins, int(base_notnull_cnt / min_sample))
q_list = [x / bin_num for x in range(1, bin_num)] 

# 得到分位数代表对应的截断值的列表
bk_list = []
for q in q_list:
    bk = base_df['score'].quantile(q)
    bk_list.append(bk)  
    
# 去重复后排序，左右两端加上 无穷 的端点值
bk_list = sorted(list(set(bk_list))) 
score_bin_list = [-np.inf] + bk_list + [np.inf] 

# 统计各分箱内的样本量
base_cnt_list = [base_null_cnt] 
test_cnt_list = [test_null_cnt] 
bucket_list = ["NAN"]

for i in range(len(score_bin_list)-1):
    left  = round(score_bin_list[i+0], 4) 
    right = round(score_bin_list[i+1], 4)
    bucket_list.append("(" + str(left) + ',' + str(right) + ']')

    base_cnt = base_df[(base_df.score > left) & (base_df.score <= right)].shape[0] # 左开右闭
    base_cnt_list.append(base_cnt)

    test_cnt = test_df[(test_df.score > left) & (test_df.score <= right)].shape[0]
    test_cnt_list.append(test_cnt)

# 汇总统计结果    
stat_df = pd.DataFrame({"bucket": bucket_list, "base_cnt": base_cnt_list, "test_cnt": test_cnt_list})
stat_df['base_dist'] = stat_df['base_cnt'] / len(base_df)
stat_df['test_dist'] = stat_df['test_cnt'] / len(test_df)

# psi 计算函数，应用到stat_df表中
def sub_psi(row):
    base_list = row['base_dist']
    test_dist = row['test_dist']
    
    # 处理某分箱内样本量为0的情况
    if base_list == 0 and test_dist == 0:
        return 0
    elif base_list == 0 and test_dist > 0:
        base_list = 1 / base_notnull_cnt   
    elif base_list > 0 and test_dist == 0:
        test_dist = 1 / test_notnull_cnt

    return (test_dist - base_list) * np.log(test_dist / base_list) #PSI的计算公式: (实际占比 - 预期占比）* ln(实际占比 / 预期占比) 。

stat_df['psi'] = stat_df.apply(lambda row: sub_psi(row), axis=1)
psi = stat_df['psi'].sum() 

