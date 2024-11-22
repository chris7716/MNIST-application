from psycopg2.extras import Json
from flask import Flask, request, jsonify, send_from_directory, Blueprint
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from flask_swagger_ui import get_swaggerui_blueprint

from service.metrics_service import evaluate_partial_metrics, fetch_metrics_from_db, evaluate_complete_metrics

# Create a blueprint for the controller
metrics_controller = Blueprint('metrics_controller', __name__)

# Swagger UI configuration
SWAGGER_URL = '/swagger'  # Swagger UI endpoint
API_URL = '/static/swagger.yaml'  # Location of the YAML file

swaggerui_blueprint = get_swaggerui_blueprint(
    SWAGGER_URL,
    API_URL,
    config={'app_name': "Model Testing and Metrics API"}
)

metrics_controller.register_blueprint(swaggerui_blueprint, url_prefix=SWAGGER_URL)

# Route to serve static files
@metrics_controller.route('/static/<path:path>')
def serve_static_files(path):
    return send_from_directory('static', path)

# Flask Route
@metrics_controller.route('/test_model', methods=['POST'])
def test_model():
    if 'model' not in request.files:
        return jsonify({'error': 'Model file is required'}), 400

    # Load Model
    model_file = request.files['model']
    
    return evaluate_complete_metrics(model_file)

# Fetch Metrics with Pagination
@metrics_controller.route('/metrics', methods=['GET'])
def get_metrics():
    page = request.args.get('page', default=None, type=int)
    page_size = request.args.get('page_size', default=10, type=int)  # Default page size is 10

    return fetch_metrics_from_db(page, page_size)

# Flask Route to Test Selected Metrics
@metrics_controller.route('/test_selected_metrics', methods=['POST'])
def test_selected_metrics():
    if 'model' not in request.files:
        return jsonify({'error': 'Model file is required'}), 400
    if 'metrics' not in request.form:
        return jsonify({'error': 'Metrics to test must be specified in the form data'}), 400

    # Extract metrics from form data
    metrics_to_test = request.form.get('metrics').split(',')
    valid_metrics = {"accuracy", "precision", "recall", "f1_score", "confusion_matrix"}
    if not all(metric.strip() in valid_metrics for metric in metrics_to_test):
        return jsonify({'error': 'Invalid metric(s) specified'}), 400

    # Load the model
    model_file = request.files['model']

    # Return the results
    return jsonify(evaluate_partial_metrics(model_file))
