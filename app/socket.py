from flask_socketio import emit
from flask import Blueprint

progress_blueprint = Blueprint("progress", __name__)
socketio = None

def init_socketio(socketio_instance):
    global socketio
    socketio = socketio_instance

    @socketio.on("connect", namespace="/progress")
    def handle_connect():
        print("Client connected to progress namespace") 

    @socketio.on("disconnect", namespace="/progress")
    def handle_disconnect():
        print("Client disconnected from progress namespace") 

def emit_progress(task_id, progress, message, sub_progress=None, sub_message=None):
    if socketio is None:
        raise RuntimeError("SocketIO instance is not initialized")

    # Prepare the progress data
    progress_data = {
        "progress": progress,
        "message": message,
        "sub_progress": sub_progress,
        "sub_message": sub_message
    }

    # Print the emitted progress message to the backend log
    print(f"Emitting progress for task {task_id}: {progress_data}")

    # Emit the progress data to the client
    socketio.emit(
        f"task_progress_{task_id}",
        progress_data,
        namespace="/progress"
    )
