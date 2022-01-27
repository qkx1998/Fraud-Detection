import numpy as np
import networkx as nx
from collections import defaultdict
import os
import matplotlib.pyplot as plt

G = nx.karate_club_graph() 

# 给节点添加标签
for node in G:
    G.add_node(node, labels = node) 
   

def slpa(G, threshold, iteration):
    """
    :param threshold:  阈值
    :param iteration:  迭代次数
    :return:
    """
    graph = G
    
    # 节点存储器初始化
    node_memory = []
    for n in range(graph.number_of_nodes()):
        node_memory.append({n+1: 1})

    # 算法迭代过程
    for t in range(iteration):
        # 任意选择一个监听器
        order = [x for x in np.random.permutation(graph.number_of_nodes())]
        for i in order:
            label_list = {}
            
            # 从speaker中选择一个标签传播到listener
            for j in graph.neighbors(i):
                sum_label = sum(node_memory[j-1].values())
                label = list(node_memory[j-1].keys())[np.random.multinomial(
                    1, [float(c) / sum_label for c in node_memory[j-1].values()]).argmax()]
                label_list[label] = label_list.setdefault(label, 0) + 1
                
            # listener选择一个最流行的标签添加到内存中
            selected_label = max(label_list, key=label_list.get)
            node_memory[i-1][selected_label] = node_memory[i-1].setdefault(selected_label, 0) + 1

    # 根据阈值threshold删除不符合条件的标签
    for memory in node_memory:
        sum_label = sum(memory.values())
        threshold_num = sum_label * threshold
        for k, v in list(memory.items()):
            if v < threshold_num:
                del memory[k]
    # 返回划分结果
    return node_memory
  
slpa(G, 0.1, 20)
    
