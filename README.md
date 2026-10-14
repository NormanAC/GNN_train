# GNN_train
try to train a GCN model by hand-writing code
## 项目简介
尝试使用GNN方法根据BDI-II的21道题目训练一个能够识别抑郁症的模型
## 项目结构
- 'read——code.py' :数据读取与简单的80%：20%测试验证
- 'gcn_knn_evaluation.py' :使用5折随机验证
- 'mlp_baseline.py':建立baseline，以求对数据泄露的验证
- 'BDI_scores.csv' :BDI问卷得分数据集
- 'README.md' :项目说明文档

## 项目历程
- 尝试读取数据进行简单处理与初步评测--read_code.py
- 使用KNN建图，GCN方法训练模型，采用5折随机验证--gnn_knn_evaluation.py
- 发现KNN数据泄露的问题（由于测试集的88人与测试集的22人存在edge_index的索引关联导致数据泄露）进行消融实验检验漏洞，因此采用MLP（多层感知机）做基线模型进行对比
- baseline的准确率有0.827，然而KNN+GCN准确率达0.945，初步确认为数据泄露

## 方向
- 修改KNN数据泄露问题
- 建立多维度判断指标（AUC、F1等）


