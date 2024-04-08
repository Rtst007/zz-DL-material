

import torch
import torch.nn as nn

# 假设 model 是你的 PyTorch 模型
model = nn.Sequential(
    nn.Linear(10, 5),
    nn.ReLU(),
    nn.Linear(5, 1)
)

# 保存整个模型
torch.save(model, 'model.pth')

dnn_input = torch.arange(start=1.0,end=10.0,step=1, dtype=torch.float32)

model_example = model()
dnn_out = model_example(dnn_input)

print(dnn_out)
a


# 加载整个模型
loaded_model = torch.load('model.pth')

# 保存模型参数
torch.save(model.state_dict(), 'model_params.pth')


# 假设 loaded_model 是你已经创建的模型的实例
loaded_model.load_state_dict(torch.load('model_params.pth'))
