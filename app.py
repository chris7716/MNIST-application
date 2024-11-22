# app.py

from flask import Flask
from controller.metrics_controller import metrics_controller
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Register the controller blueprint
app.register_blueprint(metrics_controller)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
