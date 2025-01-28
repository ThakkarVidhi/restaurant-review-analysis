from flask import Flask
from flask_socketio import SocketIO
from app.routes import api_blueprint
from app.socket import progress_blueprint, init_socketio
from config import Config

# Create a Flask app instance with template and static paths from Config
app = Flask(__name__, template_folder=Config.TEMPLATE_DIR, static_folder=Config.STATIC_DIR)

# Initialize SocketIO
socketio = SocketIO(app, cors_allowed_origins="*")

# Initialize socket handlers
init_socketio(socketio)

# Register the API blueprint
app.register_blueprint(api_blueprint)

# Register the progress blueprint
app.register_blueprint(progress_blueprint)
