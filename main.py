from app import app, socketio
from config import Config

if __name__ == "__main__":
    socketio.run(app, debug=Config.DEBUG)