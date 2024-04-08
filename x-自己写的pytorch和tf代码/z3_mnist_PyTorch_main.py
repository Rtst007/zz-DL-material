from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Normalize
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import os
import torch
import numpy as np

n_epoch = 5
BATCH_SIZE = 128
TEST_BATCH_SIZE = 1000

np.random.seed(100)

## 封装 DNN 网络
class MnistModel(nn.Module):
    def __init__(self):
        super(MnistModel, self).__init__()  # 继承
        self.fc1 = nn.Linear(1 * 28 * 28, 28)  # 参数是input和output的feature
        self.fc2 = nn.Linear(28, 10)

    def forward(self, input):
        # 1.进行形状的修改
        x = input.view([-1, 1 * 28 * 28])  # -1表示根据形状自动调整，也可以改为input.size(0)
        # 2.进行全连接的操作
        x = self.fc1(x)
        # 3.激活函数的处理
        x = F.relu(x)  # 形状没有变化
        # 4.输出层
        out = self.fc2(x)
        return F.log_softmax(out, dim=-1)

# 实例化，模型和优化器
model = MnistModel()
optimizer = Adam(model.parameters(), lr=0.001)

#######################################################################
#######################################################################
##  训练 阶段
for i in range(n_epoch):  # n_epoch 表示几轮
    transform_fn = Compose([
        ToTensor(),
        Normalize(mean=(0.1307,), std=(0.3081,))
    ])  # mean和std的形状与通道数相同

    ## 取出 训练集,  训练数据 总共有 60000 个， 6w 个
    dataset = MNIST(root='./data', train=True, transform=transform_fn)
    data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    for idx, (input, target) in enumerate(data_loader):  # idx表示data_loader中的第几个数据，元组是data_loader的数据
        optimizer.zero_grad()  # 将梯度置0
        output = model(input)  # 调用模型，得到预测值
        loss = F.nll_loss(output, target)  # 调用损失函数，得到损失,是一个tensor
        loss.backward()  # 反向传播
        optimizer.step()  # 梯度的更新
        if idx % 10 == 0:
            print(i, idx, loss.item())



#######################################################################
#######################################################################
##  测试 阶段
loss_list = []
acc_list = []

transform_fn = Compose([
    ToTensor(),
    Normalize(mean=(0.1307,), std=(0.3081,))
])  # mean和std的形状与通道数相同

## 取出 测试集
test_dataset = MNIST(root='./data', train=False, transform=transform_fn)
test_dataloader = DataLoader(test_dataset, batch_size=TEST_BATCH_SIZE, shuffle=True)

for idx, (input, target) in enumerate(test_dataloader):
    with torch.no_grad():  # 不计算梯度
        output = model(input)
        cur_loss = F.nll_loss(output, target)
        loss_list.append(cur_loss)
        # 计算准确率，output大小[batch_size,10] target[batch_size] batch_size是多少组数据，10列是每个数字概率
        pred = output.max(dim=-1)[-1]  # 获取最大值位置
        cur_acc = pred.eq(target).float().mean()
        acc_list.append(cur_acc)
print("平均准确率：", np.mean(acc_list), "平均损失：", np.mean(loss_list))

