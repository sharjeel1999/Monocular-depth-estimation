import torchvision
import torchvision.transforms as transforms

# Define the directory where you want to store the data
# Replace '/path/to/your/data' with your desired directory
data_root = "/home/sharjeel/Desktop/datasets"

# Download and load the training data
train_dataset = torchvision.datasets.MNIST(root=data_root,
                                           train=True,
                                           transform=transforms.ToTensor(), # Or other transforms
                                           download=True) 