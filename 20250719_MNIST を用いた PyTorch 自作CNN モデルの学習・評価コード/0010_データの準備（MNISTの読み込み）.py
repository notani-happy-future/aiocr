import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# 画像の前処理：Tensor化 + 正規化（0〜1 → -1〜1）
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # MNISTはグレースケール (1チャネル)
])

# データセットのダウンロードと読み込み
train_data = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_data  = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# データローダ（ミニバッチ処理）
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader  = DataLoader(test_data, batch_size=64, shuffle=False)

# データ確認（1バッチ表示）
examples = next(iter(train_loader))
images, labels = examples

plt.figure(figsize=(10, 2))
for i in range(10):
    plt.subplot(1, 10, i+1)
    plt.imshow(images[i].squeeze(), cmap="gray")
    plt.title(f"{labels[i].item()}")
    plt.axis("off")
plt.suptitle("MNIST サンプル画像")
plt.show()
