# Quickstart
## Working with data
### Types:
    1. Datasets: Samples by level, with transform arg to modify sample and target_transform to modify label
    2. Dataloaders: Iterables around Datasets and batch the data 
### Datasets:
    Torch libraries offer datasets
### Code:
    ```Python
    import torch
    from torch import nn
    from torch.utils.data import DataLoader
    from torchvision import datasets
    from torchvision.transforms import ToTensor

# Download training data from open datasets.
    training_data = datasets.FashionMNIST(
            root="data", # why?
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

# Create data loaders.
    train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

    for X, y in test_dataloader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break
    ```
## Creating Models
### Creation:
   A neural netwok inherits from nn.Module, then in the `__init__` function we define the [?] layers and in the forward function how the data pass trough the network
### Code:
   ```Python
# Get cpu, gpu or mps device for training.
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
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
print(model)
```
## Optimizing the model parameters
### Previous:
    loss function and an optimizer [?] why?
### Steps:
    1. the model makes preditions and backpropagates the error to adjust parameters
    2. check model performance against a test dataset
    3. trough epochs we see the loss and accurancy
### Code:
    ```Python 
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

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
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

        epochs = 5
        for t in range(epochs):
            print(f"Epoch {t+1}\n-------------------------------")
            train(train_dataloader, model, loss_fn, optimizer)
            test(test_dataloader, model, loss_fn)
        print("Done!")
    ```
## Saving models
serializing the model parameters
### Code:
    ```Python
    torch.save(model.state_dict(), "model.pth")
    print("Saved PyTorch Model State to model.pth")
    ```

## Loading Models
recreate the model and load the sate dict
### Code:
``` Python
model = NeuralNetwork().to(device)
model.load_state_dict(torch.load("model.pth"))


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
    x = x.to(device)
    pred = model(x)
    predicted, actual = classes[pred[0].argmax(0)], classes[y]
    print(f'Predicted: "{predicted}", Actual: "{actual}"')
```
# Tensors
Specialized data structure to encode inputs, outputs and paramters of a model
### Code:
    ```Python
        import torch
        impory numpy as np
    ```
## Intializing a Tensor
### ways:
    1. **Directly from Data** `data = [[1, 2],[3,4]]` `x_data = torch.tensor(data)`
    2. **From a Numpy Array** `np_array = np.array = np.array(data)` `x_np = torch.from_numpy(np_array)`
    3. **From another tensor**
    ```Python
    x_ones = torch.ones_like(x_data) # retains the properties of x_data
    print(f"Ones Tensor: \n {x_ones} \n")

    x_rand = torch.rand_like(x_data, dtype=torch.float) # overrides the datatype of x_data
    print(f"Random Tensor: \n {x_rand} \n")
    ```
    4. **Random Constants**
        - *Definitions*: `shape` a tuple of tensors (num_tensors, dim_tensors)
    ```Python
    shape = (2,3,)
    rand_tensor = torch.rand(shape)
    ones_tensor = torch.ones(shape)
    zeros_tensor = torch.zeros(shape)

    print(f"Random Tensor: \n {rand_tensor} \n")
    print(f"Ones Tensor: \n {ones_tensor} \n")
    print(f"Zeros Tensor: \n {zeros_tensor}")
    ```
## Attributes of a Tensor
    1. Shape
    2. datatype
    3. device (who storage)
## Operations on Tensors
Arithmetical, LA, matrix manipulation, etc..., the tensors are created in CPU so we need to specify to move Tensor to GPU `.to`
### Standard numpy-like indexing and slicing:
```Python
tensor = torch.ones(4, 4)
print(f"First row: {tensor[0]}")
print(f"First column: {tensor[:, 0]}")
print(f"Last column: {tensor[..., -1]}")
tensor[:,1] = 0
print(tensor)
```
### Joining tensors `torch.cat`
```Python
t1 = torch.cat([tensor, tensor, tensor], dim=1)
print(t1)
```
### Arithmetic
```Python
# This computes the matrix multiplication between two tensors. y1, y2, y3 will have the same value
# ``tensor.T`` returns the transpose of a tensor
y1 = tensor @ tensor.T
y2 = tensor.matmul(tensor.T)

y3 = torch.rand_like(y1)
torch.matmul(tensor, tensor.T, out=y3)


# This computes the element-wise product. z1, z2, z3 will have the same value
z1 = tensor * tensor
z2 = tensor.mul(tensor)

z3 = torch.rand_like(tensor)
torch.mul(tensor, tensor, out=z3)
```
### Single element: convert to a numerical value `item()`
```Python
agg = tensor.sum()
agg_item = agg.item()
print(agg_item, type(agg_item))
```
### Inplace operations: Operations that store results into the operand `_`
```Python
print(f"{tensor} \n")
tensor.add_(5)
print(tensor)
```
## Numpy
### Tensor to Numpy
```Python
t = torch.ones(5)
print(f"t: {t}")
n = t.numpy()
print(f"n: {n}")
```
A change in tensor will be reflected in array
```Python
t.add_(1)
print(f"t: {t}")
print(f"n: {n}")
```
### Numpy array to tensor
```Python
n = np.ones(5)
t = torch.from_numpy(n)
```
A change in array will be reflected on tensor

```Python
np.add(n, 1, out=n)
print(f"t: {t}")
print(f"n: {n}")
```
# Datasets and Data loaders
## Loading a Dataset
    Datasets attributes: root(where data is located), train(specifies data set type), download, transform and target_transform
### Code:
    ```Python
    import torch
    from torch.utils.data import Dataset
    from torchvision import datasets
    from torchvision.transforms import ToTensor
    import matplotlib.pyplot as plt


    training_data = datasets.FashionMNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor()
    )
    
    test_data = datasets.FashionMNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor()
    )
```    
## Iterate and visualize data
Datasets could be manage as a list and could use matplotlib
```Python
labels_map = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}
figure = plt.figure(figsize=(8, 8))
cols, rows = 3, 3
for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(training_data), size=(1,)).item()
    img, label = training_data[sample_idx]
    figure.add_subplot(rows, cols, i)
    plt.title(labels_map[label])
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap="gray")
plt.show()
```
## Creating a custom Dataset
Custom datasets must implement three funtions:
```Python
import os
import pandas as pd
from torchvision.io import read_image

class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
```

### `__init__` init object with the info

### `__len__` returns the number of samples

### `__getitem__` returns a sample with the index

## Preparing data with DataLoaders
```Python
from torch.utils.data import DataLoader

train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)
```

## Iterate through the DataLoader
DataLoaders returns (train_features and train_lebels) [?]
```Python
# Display image and label.
train_features, train_labels = next(iter(train_dataloader))
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")
img = train_features[0].squeeze()
label = train_labels[0]
plt.imshow(img, cmap="gray")
plt.show()
print(f"Label: {label}")
```

# Notes 
## what i learned?
1. how to manage data with pytorch
    a. Datasets: args, creation
        - Analogy: words of a book
    b. DataLoaders: args, returns
        - Analogy: Paragraphs
2. how to creaye a model
    - Analogy: Flashcards
3. How to train a model
    - Analogy: dog
4. How to save a model:
    - Analogy: taking notes
5. How to load a model:
    - Analogy: rewriting
6. Tensor management (operations, init, attr)
    - Analogy: [?]
