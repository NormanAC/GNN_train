# GNN_train
try to train a GCN model by hand-writing code
## 项目简介
尝试使用GNN方法根据BDI-II的21道题目训练一个能够识别抑郁症的模型
## 项目结构
- 'read——code.py' :数据读取与简单的80%：20%测试验证
- 'evaluation.py' :使用5折随机验证
- 'BDI_scores.csv' :BDI问卷得分数据集
- 'README.md' :项目说明文档
## 当下漏洞
- 采用KNN方法，由于测试集的88人与测试集的22人存在edge_index的索引关联导致数据泄露
