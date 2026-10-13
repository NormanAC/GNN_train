#========================================================================================================
#数据读取与处理
#========================================================================================================
import numpy as np
import pandas as pd
import random
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import torch
from sklearn.neighbors import kneighbors_graph
from torch_geometric.data import Data

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

#定义和训练评估函数
def train_and_evaluate(model,data,train_mask,test_mask):
    optimizer =torch.optim.Adam(model.parameters(),lr=0.01,weight_decay=5e-4)
    criterion =torch.nn.CrossEntropyLoss()

    data.train_mask=train_mask
    data.test_mask=test_mask

    model.train()
    for epoch in range(201):
        optimizer.zero_grad()
        out=model(data.x,data.edge_index)
        loss =criterion(out[data.train_mask],data.y[data.train_mask])
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        out=model(data.x,data.edge_index)
        pred =out.argmax(dim=1)
        test_correct=(pred[data.test_mask]==data.y[data.test_mask]).sum().item()
        test_acc=test_correct/int(data.test_mask.sum())

    return test_acc

#开始5折交叉验证

acc_list=[]
num_nodes =data.num_nodes

for fold in range(5):
    #随机切分
    indices =torch.randperm(num_nodes)
    train_size=int(num_nodes*0.8)
    train_indices =indices[:train_size]
    test_indices=indices[train_size:]
    #生成掩码
    train_mask=torch.zeros(num_nodes,dtype=torch.bool)
    train_mask[train_indices]=True
    test_mask=torch.zeros(num_nodes,dtype=torch.bool)
    test_mask[test_indices]=True

    model=GCN(in_channels=21,hidden_channels=16,out_channels=2)
    acc=train_and_evaluate(model,data,train_mask,test_mask)
    acc_list.append(acc)
    print(f'fold:{fold+1} Test Acc:{acc:.4f}')

print(f'最终成果：')
print(f'五折准确率：{np.mean(acc_list)}')


