import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from tqdm.auto import tqdm

import hivemind

# Create dataset and model, same as in the basic tutorial
# For this basic tutorial, we download only the training set
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

model = nn.Sequential(nn.Conv2d(3, 16, (5, 5)), nn.MaxPool2d(2, 2), nn.ReLU(),
                      nn.Conv2d(16, 32, (5, 5)), nn.MaxPool2d(2, 2), nn.ReLU(),
                      nn.Flatten(), nn.Linear(32 * 5 * 5, 10))
opt = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)


# Create DHT: a decentralized key-value storage shared between peers
dht = hivemind.DHT(start=True)
print("To join the training, use initial_peers =", [str(addr) for addr in dht.get_visible_maddrs()])

# Set up a decentralized optimizer that will average with peers in background
opt = hivemind.optim.DecentralizedOptimizer(
    opt,                      # wrap the SGD optimizer defined above
    dht,                      # use a DHT that is connected with other peers
    average_parameters=True,  # periodically average model weights in opt.step
    average_gradients=False,  # do not average accumulated gradients
    prefix='my_cifar_run',    # unique identifier of this collaborative run
    target_group_size=16,     # maximum concurrent peers for this run
    verbose=True              # print logs incessently
)
# Note: if you intend to use GPU, switch to it only after the decentralized optimizer is created

with tqdm() as progressbar:
    while True:
        for x_batch, y_batch in torch.utils.data.DataLoader(trainset, shuffle=True, batch_size=256):
            opt.zero_grad()
            loss = F.cross_entropy(model(x_batch), y_batch)
            loss.backward()
            opt.step()

            progressbar.desc = f"loss = {loss.item():.3f}"
            progressbar.update()