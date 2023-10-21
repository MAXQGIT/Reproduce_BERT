import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'  # 设备
print(device)


print(torch.cuda.is_available())