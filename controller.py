from flask import Flask, request, jsonify
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import numpy as np
import io

app = Flask(__name__)

# Define the model structure (same as before)
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(28*28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# Define the transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Helper function to evaluate the model and calculate metrics
def evaluate_model(model, test_loader):
    model.eval()  # Set the model to evaluation mode
    all_labels = []
    all_predictions = []

    with torch.no_grad():  # No need to compute gradients during evaluation
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    # Calculate accuracy
    accuracy = 100 * np.sum(np.array(all_labels) == np.array(all_predictions)) / len(all_labels)
    
    # Calculate precision, recall, and f1-score
    precision = precision_score(all_labels, all_predictions, average='weighted')
    recall = recall_score(all_labels, all_predictions, average='weighted')
    f1 = f1_score(all_labels, all_predictions, average='weighted')
    
    # Compute confusion matrix
    conf_matrix = confusion_matrix(all_labels, all_predictions)

    return accuracy, precision, recall, f1, conf_matrix

@app.route('/test_model', methods=['POST'])
def test_model():
    # Check if the model is provided in the request
    if 'model' not in request.files:
        return jsonify({'error': 'Model is required'}), 400

    model_file = request.files['model']

    # Load the model from the uploaded file
    model = SimpleNN()
    model.load_state_dict(torch.load(model_file))
    
    # Load MNIST test dataset
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Evaluate the model and get the metrics
    accuracy, precision, recall, f1, conf_matrix = evaluate_model(model, test_loader)
    
    # Return the metrics as JSON response
    return jsonify({
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': conf_matrix.tolist()  # Convert matrix to list for JSON compatibility
    })

if __name__ == '__main__':
    app.run(debug=True)
