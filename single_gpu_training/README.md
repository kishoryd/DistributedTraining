

# **Single-GPU Training**

Optimizing your script for a single GPU is a crucial first step before scaling to multiple GPUs. Inefficient single-GPU code may result in wasted resources and longer queue times when running multi-GPU jobs. This example demonstrates how to train a CNN on the MNIST dataset using a single GPU and profile the training process for performance improvements.

```bash
$ ssh <hpcuser>@14.139.62.247 -p 4422
$ git clone https://github.com/kishoryd/DistributedTraining.git
```

## **Step 1: Activate Conda Environment**

To simplify the setup, use a pre-configured Conda environment. Follow these steps:

```bash
$ module load miniconda
$ conda activate gujcost_workshop
```
## **Step 2: Dowload the MNIST dataset**

Ensure the dataset is downloaded on the login node since compute nodes typically lack internet access:

```bash
(gujcost_workshop) $ python download_data.py
```

## **Step 3: Run and Profile the Script**

First, inspect the training script (`mnist_model.py`) by navigating to its directory and viewing its content:

```bash
(gujcost_workshop) $ cd path/to/single_gpu_training
(gujcost_workshop) $ cat mnist_model.py
```
Analyse the code for better understanding
```
# Importing required libraries
import argparse  # For handling command-line arguments
import torch  # PyTorch library for tensor operations
import torch.nn as nn  # PyTorch modules for neural network layers
import torch.nn.functional as F  # Functions for activations, losses, etc.
import torch.optim as optim  # PyTorch optimization algorithms
import os  # For accessing environment variables and file handling
from torchvision import datasets, transforms  # For data handling and preprocessing
from torch.optim.lr_scheduler import StepLR  # Learning rate scheduler

# Define the CNN model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # First convolutional layer: input channels = 1, output channels = 32, kernel size = 3
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        # Second convolutional layer: input channels = 32, output channels = 64, kernel size = 3
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        # Dropout layers to prevent overfitting
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        # Fully connected layers: 9216 input features, 128 output features
        self.fc1 = nn.Linear(9216, 128)
        # Output layer: 128 input features, 10 output features (for 10 MNIST classes)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # Forward pass through the first convolutional layer and activation
        x = self.conv1(x)
        x = F.relu(x)
        # Forward pass through the second convolutional layer and activation
        x = self.conv2(x)
        x = F.relu(x)
        # Apply max pooling with kernel size 2
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)  # Apply dropout
        x = torch.flatten(x, 1)  # Flatten the feature map into a 1D tensor
        x = self.fc1(x)  # Fully connected layer
        x = F.relu(x)
        x = self.dropout2(x)  # Apply dropout
        x = self.fc2(x)  # Output layer
        # Apply log softmax for classification probabilities
        output = F.log_softmax(x, dim=1)
        return output

# Training loop
try:
    from line_profiler import profile  # Importing line_profiler for profiling
except ImportError:
    def profile(func):  # Fallback if @profile is not available
        return func

@profile
def train(args, model, device, train_loader, optimizer, epoch):
    model.train()  # Set the model to training mode
    for batch_idx, (data, target) in enumerate(train_loader):  # Loop through batches of data
        # Move data and target labels to the specified device (CPU or GPU)
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()  # Reset gradients from the previous step
        output = model(data)  # Forward pass
        loss = F.nll_loss(output, target)  # Compute loss (negative log likelihood)
        loss.backward()  # Backpropagation to compute gradients
        optimizer.step()  # Update model parameters using gradients
        # Log training progress every `log_interval` batches
        if batch_idx % args.log_interval == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
            if args.dry_run:  # Stop early if dry run is enabled
                break

# Testing loop
def test(model, device, test_loader):
    model.eval()  # Set the model to evaluation mode
    test_loss = 0
    correct = 0
    with torch.no_grad():  # Disable gradient computation for testing
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)  # Move data to the specified device
            output = model(data)  # Forward pass
            # Sum the test loss across the entire test set
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            # Get predictions by selecting the class with the highest probability
            pred = output.argmax(dim=1, keepdim=True)
            # Count the number of correct predictions
            correct += pred.eq(target.view_as(pred)).sum().item()
    # Compute average loss and accuracy
    test_loss /= len(test_loader.dataset)
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} '
          f'({100. * correct / len(test_loader.dataset):.0f}%)\n')

# Main function
def main():
    # Parse command-line arguments for configurable options
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()

    # Set device to CUDA if available and not disabled
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)  # Set the random seed for reproducibility
    device = torch.device("cuda" if use_cuda else "cpu")

    # Configure DataLoader parameters
    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    num_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))  # Use Slurm environment variable
    if use_cuda:
        cuda_kwargs = {'num_workers': num_workers, 'pin_memory': True, 'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    # Define dataset transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # Normalize MNIST images
    ])

    # Load MNIST dataset
    try:
        dataset1 = datasets.MNIST('./data', train=True, download=False, transform=transform)
        dataset2 = datasets.MNIST('./data', train=False, download=False, transform=transform)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    print(f"Train set size: {len(dataset1)}")
    print(f"Test set size: {len(dataset2)}")

    # Initialize model, optimizer, and scheduler
    model = Net().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

    # Train and test the model
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
        scheduler.step()  # Adjust learning rate

    # Save the trained model if specified
    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")

if __name__ == '__main__':
    main()
```

### **Decorate the Training Function for Profiling**
Add a `line_profiler` decorator to the `train` function:

```python
@profile
def train(args, model, device, train_loader, optimizer, epoch):
    ...
```

### **Slurm Script for Single-GPU Training**

Create a `slurm_submit.sh` file with the following content:

```bash#!/bin/bash
#SBATCH --job-name=mnist_1GPU    # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --partition=gpu          # Specify GPU partition for GPU Nodes
##SBATCH --reservation=hpcai      # Reservation incase of urgent requirement
##SBATCH --nodelist=rpgpu*        # Specify reservation GPU node name provided
#SBATCH --gres=gpu:1             # number of gpus per node
#SBATCH --output=logs_%j.out     # output logfile name
#SBATCH --error=logs_%j.err      # error logfile name
#SBATCH --time=00:10:00          # total run time limit (HH:MM:SS)

# which gpu node was used
echo "Running on host of RUDRA" $(hostname)

module purge #remove unneccesary loaded modules

module load miniconda

conda activate gujcost_workshop

kernprof -o ${SLURM_JOBID}_${SLURM_CPUS_PER_TASK}.lprof -l mnist_model.py --epochs=5
```

## **Step 4: Submit job to GPU Node**

```bash
(gujcost_workshop) $ sbatch slurm_submit.sh
```

The job will execute, and the results will be logged.


## **Step 5: Analyze Profiling Data**

After the job completes, analyze the profiling results:

```bash
(gujcost_workshop) $ python -m line_profiler -rmt logs_<job_id>.lprof
```

Example output:

```
Timer unit: 1e-06 s

Total time: 93.0049 s
File: mnist_model.py
Function: train at line 39

Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
    39                                           @profile
    40                                           def train(args, model, device, train_loader, optimizer, epoch):
    ...
```

The profiling data highlights bottlenecks in the training loop, such as data transfer or model computation.


### **Update Slurm Settings**
Change the Slurm script to utilize more CPU cores:

```bash
#SBATCH --cpus-per-task=4
```
```bash
#SBATCH --cpus-per-task=8
```
Submit the job again and observe the performance improvement.


## **Summary**

Optimizing single-GPU training is a critical step before scaling to multiple GPUs. Efficient code ensures reduced resource usage and shorter queue times. By profiling and addressing bottlenecks, you can achieve higher GPU utilization and better performance.


## **How the Conda Environment Was Created**
If needed, create the Conda environment with these commands:

```bash
$ module load miniconda
$ conda create --name gujcost_workshop python=3.9 --yes
$ conda activate gujcost_workshop
$ pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
$ conda install line_profiler --channel conda-forge
```
