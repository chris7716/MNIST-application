import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# Define the neural network model
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(28*28, 128)  # Input layer (28x28 pixels)
        self.fc2 = nn.Linear(128, 64)     # Hidden layer
        self.fc3 = nn.Linear(64, 10)      # Output layer (10 classes)

    def forward(self, x):
        x = x.view(-1, 28*28)  # Flatten the image
        x = torch.relu(self.fc1(x))  # ReLU activation
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Define the training and testing transformations
transform = transforms.Compose([
    transforms.ToTensor(),  # Convert images to tensors
    transforms.Normalize((0.5,), (0.5,))  # Normalize the images
])

# Load the MNIST dataset
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Initialize the model, loss function, and optimizer
model = SimpleNN()
criterion = nn.CrossEntropyLoss()  # Cross-entropy loss for multi-class classification
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
num_epochs = 5

for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    running_loss = 0.0
    img_count = 0

    for images, labels in train_loader:
        optimizer.zero_grad()  # Zero the gradients

        outputs = model(images)  # Forward pass
        loss = criterion(outputs, labels)  # Compute the loss

        loss.backward()  # Backward pass
        optimizer.step()  # Optimize the weights

        running_loss += loss.item()
        
        img_count += 1

    print(f"Img count {img_count}")
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}")

# Evaluate the model on the test set
model.eval()  # Set the model to evaluation mode
correct = 0
total = 0

with torch.no_grad():  # No need to compute gradients during evaluation
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f'Test Accuracy: {accuracy}%')

# Save the trained model
torch.save(model.state_dict(), "mnist_model.pth")
print("Model saved to mnist_model.pth")

# Visualize some test images and their predictions
def imshow(img):
    img = img / 2 + 0.5  # Unnormalize
    plt.imshow(img.numpy().squeeze(), cmap='gray')
    plt.show()

# Visualizing some test images
data_iter = iter(test_loader)
images, labels = next(data_iter)

# Display first few images and predictions
outputs = model(images)
_, predicted = torch.max(outputs, 1)

for i in range(5):
    imshow(images[i])
    print(f"Predicted: {predicted[i].item()}, Actual: {labels[i].item()}")
