import nni
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
# from torchvision import datasets
# from torchvision.transforms import ToTensor
#import os
#os.environ['CUDA_VISIBLE_DEVICES']="0"

params = {
        'features': 512,
        'lr': 0.001,
        'momentum': 0,
        'epochs': 5,
}

optimized_params = nni.get_next_parameter()
params.update(optimized_params)

X = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], dtype=torch.float32)
y = torch.tensor([0, 1, 1, 0], dtype=torch.float32)


# Create data loaders.
dataset = TensorDataset(X, y.unsqueeze(1))
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

# Get cpu, gpu or mps device for training.
device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)
#device = 'cpu'
print(f"Using {device} device")

# Define model
class XORNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(2, params['features']), 
            nn.ReLU(),
            nn.Linear(params['features'], params['features']), 
            nn.ReLU(),
            nn.Linear(params['features'], 1),  
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.linear_relu_stack(x) 

model = XORNet().to(device)
print(model)

loss_fn = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=params['lr'], momentum=params['momentum'])

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


epochs = params['epochs']
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(dataloader, model, loss_fn, optimizer)


final_loss = loss_fn(model(X.to(device)), y.unsqueeze(1).to(device))
nni.report_final_result(final_loss.item())
print("Training completed")