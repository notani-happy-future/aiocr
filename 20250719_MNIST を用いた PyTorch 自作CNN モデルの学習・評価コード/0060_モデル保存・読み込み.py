# 保存
torch.save(model.state_dict(), 'mnist_cnn.pth')

# 読み込み
# model = SimpleCNN()
# model.load_state_dict(torch.load('mnist_cnn.pth'))
# model.to(device)
