import os
import psycopg2
from psycopg2.extras import Json
from flask import Flask, request, jsonify
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Flask App Initialization
app = Flask(__name__)

# Database Connection
DB_HOST = "localhost"
DB_NAME = "postgres"
DB_USER = "postgres"
DB_PASSWORD = "test123"
DB_PORT = 5432

def get_db_connection():
    return psycopg2.connect(
        host=DB_HOST,
        database=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        port=DB_PORT
    )

# Model Definition
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Load MNIST Test Dataset
transform = transforms.Compose([transforms.ToTensor()])
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Evaluate Model
def evaluate_model(model, test_loader):
    model.eval()
    predictions, true_labels = [], []

    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            predictions.extend(preds.numpy())
            true_labels.extend(labels.numpy())

    # Calculate Metrics
    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions, average='weighted')
    recall = recall_score(true_labels, predictions, average='weighted')
    f1 = f1_score(true_labels, predictions, average='weighted')
    conf_matrix = confusion_matrix(true_labels, predictions)
    
    # Convert NumPy types to native Python types
    accuracy = float(accuracy)
    precision = float(precision)
    recall = float(recall)
    f1 = float(f1)

    return accuracy, precision, recall, f1, conf_matrix

# Flask Route
@app.route('/test_model', methods=['POST'])
def test_model():
    if 'model' not in request.files:
        return jsonify({'error': 'Model file is required'}), 400

    # Load Model
    model_file = request.files['model']
    model_path = os.path.join('./', model_file.filename)
    model_file.save(model_path)
    model = SimpleNN()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

    # Evaluate Model
    accuracy, precision, recall, f1, conf_matrix = evaluate_model(model, test_loader)

    # Convert confusion matrix to list (for database compatibility)
    conf_matrix_list = conf_matrix.tolist()

    # Store Metrics in PostgreSQL
    connection = get_db_connection()
    cursor = connection.cursor()
    conf_mat = Json(conf_matrix_list)
    try:
        cursor.execute(
            """
            INSERT INTO model_metrics (model_name, accuracy, precision, recall, f1_score, confusion_matrix)
            VALUES (%s, %s, %s, %s, %s, %s)
            """,
            ('mnist_model', accuracy, precision, recall, f1, conf_mat)
        )
        connection.commit()
    except Exception as e:
        connection.rollback()
        print(e)
        return jsonify({'error': f'Database error: {str(e)}'}), 500
    finally:
        cursor.close()
        connection.close()

    # Return Metrics
    return jsonify({
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': conf_matrix_list
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
