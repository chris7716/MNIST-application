import os
import psycopg2
import torch
from psycopg2.extras import Json
from flask import jsonify
from dotenv import load_dotenv
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

load_dotenv()

# Database Connection
DB_HOST = os.getenv("DB_HOST", "xxxxx")
DB_NAME = os.getenv("DB_NAME", "xxxxx")
DB_USER = os.getenv("DB_USER", "xxxxx")
DB_PASSWORD = os.getenv("DB_PASSWORD", "xxxxx")
DB_PORT = int(os.getenv("DB_PORT", 1111))

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

# Evaluate Model with Selected Metrics
def evaluate_selected_metrics(model, data_loader, metrics_to_test):
    predictions, true_labels = [], []

    with torch.no_grad():
        for images, labels in data_loader:
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            predictions.extend(preds.numpy())
            true_labels.extend(labels.numpy())

    results = {}
    if "accuracy" in metrics_to_test:
        results["accuracy"] = float(accuracy_score(true_labels, predictions))
    if "precision" in metrics_to_test:
        results["precision"] = float(precision_score(true_labels, predictions, average='weighted'))
    if "recall" in metrics_to_test:
        results["recall"] = float(recall_score(true_labels, predictions, average='weighted'))
    if "f1_score" in metrics_to_test:
        results["f1_score"] = float(f1_score(true_labels, predictions, average='weighted'))
    if "confusion_matrix" in metrics_to_test:
        results["confusion_matrix"] = confusion_matrix(true_labels, predictions).tolist()

    return results

def evaluate_partial_metrics(model_file):
    # Load Model
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
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': conf_matrix_list
    }

def evaluate_complete_metrics(model_file):
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

def fetch_metrics_from_db(page, page_size):
    connection = get_db_connection()
    cursor = connection.cursor()

    try:
        if page:
            # Paginated Fetch
            offset = (page - 1) * page_size
            cursor.execute(
                """
                SELECT model_name, accuracy, precision, recall, f1_score, confusion_matrix 
                FROM model_metrics 
                ORDER BY id DESC 
                LIMIT %s OFFSET %s
                """, 
                (page_size, offset)
            )
        else:
            # Fetch All
            cursor.execute(
                """
                SELECT model_name, accuracy, precision, recall, f1_score, confusion_matrix 
                FROM model_metrics 
                ORDER BY id DESC
                """
            )

        rows = cursor.fetchall()
        metrics = [
            {
                "model_name": row[0],
                "accuracy": row[1],
                "precision": row[2],
                "recall": row[3],
                "f1_score": row[4],
                "confusion_matrix": row[5]
            }
            for row in rows
        ]

        if not metrics:
            return jsonify({'message': 'No metrics found.'}), 404

        return jsonify(metrics)
    except Exception as e:
        return jsonify({'error': f'Database error: {str(e)}'}), 500
    finally:
        cursor.close()
        connection.close()