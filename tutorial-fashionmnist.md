# Running Tutorials

Once you have an environment setup, you can run the [tutorials from pytorch.org](https://pytorch.org/tutorials/beginner/basics/intro.html) locally

## Checking your environment

Make sure you are in the correct conda environment. If you followed the [Install document](install.md), then you should be using `torch` as your conda environment. Switch to `torch` with:

```shell
conda activate torch
```

To make it easier, you can run the following code in a jupyter notebook by running the following:

```shell
jupyter notebook
```

As a sanity check, run the following code to make sure you are using the correct environment and using the GPU

```python
import sys
import torch

print(f"PyTorch Version: {torch.__version__}")
print()
print(f"Python {sys.version}")
print("GPU is", "available" if torch.cuda.is_available() else "NOT AVAILABLE")
```

## Quick Start Tutorial

The [Quick Start tutorial](https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html) is a great way of checking your system and seeing how it performs.

1. Get imports and Download training/test data from the Fashion MNIST data set. This may take a while depending on your network speed.  

    ```python
    import torch
    from torch import nn
    from torch.utils.data import DataLoader
    from torchvision import datasets
    from torchvision.transforms import ToTensor
    
    # Download training data from open datasets.
    training_data = datasets.FashionMNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor(),
    )
    
    # Download test data from open datasets.
    test_data = datasets.FashionMNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor(),
    )
    ```

2. Set batch size, we break our data into batches since it can be huge. We also want to load our data

    ```python
    batch_size = 64

    # Create data loaders
    train_dataloader = DataLoader(training_data, batch_size=batch_size)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)
    
    for X, y in test_dataloader:
        print(f"Shape of X [N, C, H, W]: {X.shape}")
        print(f"Shape of y: {y.shape} {y.dtype}")
        break
    ```

    You should see something similar to:

    ```shell
    Shape of X [N, C, H, W]: torch.Size([64, 1, 28, 28])
    Shape of y: torch.Size([64]) torch.int64
    ```

    > Notice that from X, Height and Width are 28  

    The images in the Fashion MNIST data set are cropped to 28x28, to learn more go to the [Fashion MNIST GitHub](https://github.com/zalandoresearch/fashion-mnist)  

3. We now want to determine if we should run on the GPU(cuda) or CPU

    ```python
    # Get cpu or gpu device for training.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")
    ```

    You should see something similar to:

    ```shell
    Using cuda device
    ```

4. Now define a model to be used and move it to the device, in our case it should be GPU

    ```python
    # Define model
    class NeuralNetwork(nn.Module):
        def __init__(self):
            super(NeuralNetwork, self).__init__()
            self.flatten = nn.Flatten()
            self.linear_relu_stack = nn.Sequential(
                # note here that 28*28 refers to the Height and Width from above
                nn.Linear(28*28, 512),
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, 10)
            )
    
        def forward(self, x):
            x = self.flatten(x)
            logits = self.linear_relu_stack(x)
            return logits
    
    model = NeuralNetwork().to(device)
    # print the model so we know what we are looking at
    # in_features=784 corresponds to 28*28 which is the flattened version of 28*28 of our image
    print(model)
    ```

    You should see something similar to:

    ```shell
    NeuralNetwork(
      (flatten): Flatten(start_dim=1, end_dim=-1)
      (linear_relu_stack): Sequential(
        (0): Linear(in_features=784, out_features=512, bias=True)
        (1): ReLU()
        (2): Linear(in_features=512, out_features=512, bias=True)
        (3): ReLU()
        (4): Linear(in_features=512, out_features=10, bias=True)
      )
    )    
    ```

5. Define a way of Training and Testing the current model

    ```python
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    
    def train(dataloader, model, loss_fn, optimizer):
        size = len(dataloader.dataset)
        model.train()
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
            
            # Compute prediction error
            pred = model(X)
            loss = loss_fn(pred, y)
            
            # Back Propagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if batch % 100 == 0:
                loss, current = loss.item(), batch * len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    def test(dataloader, model, loss_fn):
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        model.eval()
        test_loss, correct = 0, 0
        with torch.no_grad():
            for X, y in dataloader:
                X, y = X.to(device), y.to(device)
                pred = model(X)
                test_loss += loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        test_loss /= num_batches
        correct /= size
        print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    ```

6. Train on the data set! This step can take a while depending on number of epochs chosen.

    > Note: you should see the test **Accuracy Go Up** and **Avg Loss Go Down** each epoch.  Something neat to try is **changing the number of epochs** to see if you can **improve the accuracy**

    ```python
    import time
    start_time = time.time()

    print("Begin Training...")
    epochs = 5
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer)
        test(test_dataloader, model, loss_fn)
    print("Finished Training... took ", time.time() - start_time, " secs")
    ```

7. In order to re-use your model, you can save it:

    ```python
    torch.save(model.state_dict(), "model.pth")
    print("Saved PyTorch Model State to model.pth")
    ```

8. Once you save your model, you can load it with:

    ```python
    model = NeuralNetwork()
    model.load_state_dict(torch.load("model.pth"))
    ```

9. Once we load the model, we can use it to make predictions.

    > Note that the labels for the images are specific to the Fashion MNIST data set. To learn more go to the [Fashion MNIST GitHub](https://github.com/zalandoresearch/fashion-mnist) and look at the section called [Labels](https://github.com/zalandoresearch/fashion-mnist#labels)

    ```python
    classes = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
    ]

    model.eval()
    x, y = test_data[0][0], test_data[0][1]
    with torch.no_grad():
        pred = model(x)
        predicted, actual = classes[pred[0].argmax(0)], classes[y]
        print(f'Predicted: "{predicted}", Actual: "{actual}"')
    ```

    You should see something similar to:

    ```shell
    Predicted: "Ankle boot", Actual: "Ankle boot"
    ```

    If your `Predicted` doesn't match the `Actual`, you can increase the number of epochs to get your accuracy to go up.

## Playing with CPU

In order to test the difference between CPU and GPU Acceleration, you can run the following code:

```python
# Use the CPU
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

# Download training data from open datasets.
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

# Download test data from open datasets.
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)

batch_size = 64

# Create data loaders
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

for X, y in test_dataloader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break

# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            # note here that 28*28 refers to the Height and Width from above
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork()
# print the model so we know what we are looking at
# in_features=784 corresponds to 28*28 which is the flattened version of 28*28 of our image
print(model)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):      
        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)
        
        # Back Propagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

import time
start_time = time.time()

print("Begin Training...")
epochs = 5
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
print("Finished Training... took ", time.time() - start_time, " secs")

classes = [
"T-shirt/top",
"Trouser",
"Pullover",
"Dress",
"Coat",
"Sandal",
"Shirt",
"Sneaker",
"Bag",
"Ankle boot",
]

model.eval()
x, y = test_data[0][0], test_data[0][1]
with torch.no_grad():
    pred = model(x)
    predicted, actual = classes[pred[0].argmax(0)], classes[y]
    print(f'Predicted: "{predicted}", Actual: "{actual}"')

```

## Conclusion

That concludes the intro tutorial from pytorch.org.
I've included several other tutorials in the [README.md](README.md) file if you want to learn more.

I can be reached at @IAmDanielV on Twitter if you have any questions or suggestions.

Thanks!
