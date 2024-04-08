##  鸢尾花 分类

# 步骤 1：加载必要的库
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np


n_epoch = 10001

# 设置随机种子
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(seed)


# 步骤 2：加载数据集
iris = datasets.load_iris()
x_train, x_test, y_train, y_test = train_test_split(
iris.data, iris.target, test_size=0.2, random_state=42
)   # random_state表示随机数种子，用于划分测试集和训练集

# 转换 格式
x_train = torch.tensor(x_train, dtype=torch.float32)
x_test = torch.tensor(x_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.int64)
y_test = torch.tensor(y_test, dtype=torch.int64)


# 步骤 3：构建 PyTorch 模型
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(4, 3)

    def forward(self, x):
        return torch.softmax(self.fc(x), dim=1)


model = SimpleModel()

# 步骤 4：定义损失函数和优化器
optimizer = optim.SGD(model.parameters(), lr=0.1, weight_decay=1e-4) #  weight_decay=1e-4是正则化项

# 步骤 5：训练模型
# x_train, y_train = torch.from_numpy(x_train).float(), torch.from_numpy(y_train)
for epoch in range(1, n_epoch):
    model.train()         # 设置为训练模式

    outputs = model(x_train)  # 前向传播
    loss = torch.nn.functional.cross_entropy(outputs, y_train)  # 计算损失


    optimizer.zero_grad()   # 清空过往梯度,  在每次的 loss.backward() 之前使用
    loss.backward()         # 反向传播，计算梯度
    optimizer.step()        # 参数更新

    if epoch % 100 == 0:
        print(f'Epoch {epoch}/{500}, Loss: {loss.item()}')

# 步骤 6：打印模型摘要
# print(model)


# 步骤 7：在测试集上评估模型
model.eval()  # 将模型设置为评估模式

# 获取模型预测结果
with torch.no_grad():  # 在评估模式下不需要计算梯度
    outputs = model(x_test)

# 计算准确率
predicted_labels = torch.argmax(outputs, dim=1)
accuracy = (predicted_labels == y_test).sum().item() / len(y_test)
print(f'Test Accuracy: {accuracy * 100:.2f}%')