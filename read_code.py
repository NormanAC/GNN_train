#========================================================================================================
#数据读取与处理
#========================================================================================================
import numpy as np
import pandas as pd
import random
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import torch

seed=42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

df=pd.read_csv('BDI_scores.csv')

features_columns=[col for col in df.columns if col not in ['participant_id','BDI_total']]
x_array =df[features_columns].values
print({x_array.shape})

#二分类
y_array =(df['BDI_total']>13).astype(int).values
print(f'标签维度{y_array.shape}')
print(f'标签分布{np.bincount(y_array)}')

#转化为pytorch张量
x= torch.tensor(x_array,dtype=torch.float32)
y= torch.tensor(y_array,dtype=torch.long)
print(f'x的形状:{x.shape},类型:{x.dtype}')
print(f'y的形状:{y.shape},类型:{y.dtype}')

#图形化
from sklearn.neighbors import kneighbors_graph
from torch_geometric.data import Data
k=5
adj_matrix =kneighbors_graph(x_array,n_neighbors=k,mode='connectivity',include_self=False)
adj_coo=adj_matrix.tocoo()
edge_index=torch.tensor(np.vstack((adj_coo.row,adj_coo.col)),dtype=torch.long)

#edge_index形状
print(edge_index.shape)
data=Data(x=x,edge_index=edge_index,y=y)

#图数据Data对象
print(data)

#======================================================================================================
#模型定义与训练循环
#======================================================================================================

print('定义GCN模型')
class GCN(torch.nn.Module):
    def __init__(self,in_channels,hidden_channels,out_channels):
        super().__init__()
        self.conv1 =GCNConv(in_channels,hidden_channels)
        self.conv2 =GCNConv(hidden_channels,out_channels)

    def forward(self,x,edge_index):
        x=self.conv1(x,edge_index)
        x=F.relu(x)
        x=F.dropout(x,p=0.5,training=self.training)
        x=self.conv2(x,edge_index)
        return x

#实例化模型
model=GCN(in_channels=21,hidden_channels=16,out_channels=2)
print(model)

#划分训练集和测试集---80%用来训练，20%用来测试
indices=torch.randperm(data.num_nodes)
train_size=int(data.num_nodes*0.8)
train_indices=indices[:train_size]
test_indices=indices[train_size:]
#生成布尔掩码
data.train_mask=torch.zeros(data.num_nodes,dtype=torch.bool)
data.train_mask[train_indices]=True
data.test_mask=torch.zeros(data.num_nodes,dtype=torch.bool)
data.test_mask[test_indices]=True

print(f'训练集节点数：{data.train_mask.sum().item()}')
print(f'测试集节点数：{data.test_mask.sum().item()}')

#定义优化器和损失函数
optimizer=torch.optim.Adam(model.parameters(),lr=0.01,weight_decay=5e-4)
criterion=torch.nn.CrossEntropyLoss()

#开始循环训练
model.train()
for epoch in range(201):
    optimizer.zero_grad()
    out=model(data.x,data.edge_index)

    loss=criterion(out[data.train_mask],data.y[data.train_mask])
    loss.backward()#反向传播
    optimizer.step()#更新参数

    if epoch%50==0:
        print(f'epoch:{epoch:03d},loss:{loss:.4f}')

#最终测试集评估
model.eval()#切换到评估模式
with torch.no_grad():
    out=model(data.x,data.edge_index)
    pred=out.argmax(dim=1)
    test_correct=pred[data.test_mask]==data.y[data.test_mask]
    test_acc=int(test_correct.sum())/int(data.test_mask.sum())

    train_correct=pred[data.train_mask]==data.y[data.train_mask]
    train_acc=int(train_correct.sum())/int(data.train_mask.sum())

    print(f'训练集准确率：{train_acc:.4f}')
    print(f'测试集准确率：{test_acc:.4f}')




