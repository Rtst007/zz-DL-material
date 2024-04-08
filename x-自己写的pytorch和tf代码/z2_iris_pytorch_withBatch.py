##  鸢尾花 分类

# 步骤 1：加载必要的库
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np

n_epoch = 10001
batch_size = 32

# 设置随机种子
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(seed)

# 步骤 2：加载数据集   数据集 总共有  150个鸢尾花图片数据， test_size=0.2，因此训练集120， 测试集30
iris = datasets.load_iris()
x_train, x_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42
)

# print(x_train.shape)
# a

# 转换格式
x_train = torch.tensor(x_train, dtype=torch.float32)
x_test = torch.tensor(x_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.int64)  # 训练 标签
y_test = torch.tensor(y_test, dtype=torch.int64)    # 测试 标签


# 将数据封装为 TensorDataset
train_dataset = TensorDataset(x_train, y_train)  # 将训练数据张量 和 训练标签张量 打包
test_dataset = TensorDataset(x_test, y_test)


# 使用 DataLoader 进行批量训练
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

# 步骤 3：构建 PyTorch 模型
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(4, 3)

    def forward(self, x):
        return torch.softmax(self.fc(x), dim=1)

model = SimpleModel()

# 步骤 4：定义损失函数和优化器
optimizer = optim.SGD(model.parameters(), lr=0.1, weight_decay=1e-4)

# 步骤 5：训练模型
for epoch in range(1, n_epoch):
    model.train()         # 设置为训练模式

    for x_batch, y_batch in train_loader:  # 进行 batch 级别的循环
        outputs = model(x_batch)  # 前向传播
        loss = torch.nn.functional.cross_entropy(outputs, y_batch)  # 计算损失

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
