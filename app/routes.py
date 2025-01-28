import os
from datetime import datetime
import threading
from flask import Blueprint, request, jsonify, render_template
import pandas as pd
from app.scrapper import scrape_reviews
from app.sentiment_analysis import (
    load_model,
    preprocess_data,
    extract_advanced_features,
    assign_sentiment_label,
    predict_sentiment,
    create_dashboard,
    save_analysis_data
)
from config import Config
from app.socket import emit_progress

# Blueprint for API routes
api_blueprint = Blueprint("api", __name__)

# Dictionary to store task results
task_results = {}

@api_blueprint.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@api_blueprint.route("/analyze", methods=["POST"])
def analyze():
    try:
        restaurant_input = request.form.get("restaurant").replace(" ", "-")
        if not restaurant_input:
            return jsonify({"error": "Restaurant input is required!"}), 400

        # Generate a unique task ID
        restaurant_generic_name = restaurant_input.replace("-", "_")
        current_timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        task_id = f"task--{os.urandom(4).hex()}--{restaurant_generic_name}--{current_timestamp}"

        # Start the analysis task in a background thread
        task_thread = threading.Thread(target=analyze_task, args=(restaurant_input, task_id))
        task_thread.start()

        # Immediately return the task ID to the user
        return jsonify({"message": "Analysis started", "task_id": task_id})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def analyze_task(restaurant_name, task_id):
    try:
        # Emit initial progress
        emit_progress(task_id, 10, "Initializing analysis...")

        # Construct the URL
        url = Config.SCRAPPING_BASE_URL + restaurant_name

        # Step 1: Scrape Reviews
        emit_progress(task_id, 20, "Scraping reviews...")
        reviews_csv_file_name = scrape_reviews(url, Config.CHROME_DRIVER_PATH, task_id) 
        reviews_csv_file = os.path.join(Config.RAW_DATA_DIR, reviews_csv_file_name)
        print(reviews_csv_file)

        if not os.path.exists(reviews_csv_file):
            emit_progress(task_id, 100, "No reviews found!")
            return

        # Step 2: Load Data
        emit_progress(task_id, 30, "Loading data...")
        data = pd.read_csv(reviews_csv_file)

        # Step 3: Preprocess Data
        emit_progress(task_id, 40, "Preprocessing data...")
        data = preprocess_data(data)
        print(data.head())
        print(data.columns)

        # Step 4: Extract Features
        emit_progress(task_id, 50, "Extracting features...")
        features_df = data.apply(lambda x: extract_advanced_features(x['cleaned_comment'], x['city']), axis=1)
        data = pd.concat([data, features_df], axis=1)
        print(data.head())
        print(data.columns)

        # Step 5: Assign Sentiment Labels
        emit_progress(task_id, 60, "Assigning sentiment labels...")
        data["sentiment_label"] = data.apply(assign_sentiment_label, axis=1)
        print(data.head())
        print(data.columns)

        # Step 6: Load Trained Model
        emit_progress(task_id, 70, "Loading trained model...")
        model = load_model(Config.MODEL_PATH)
        print(model)

        # Step 7: Perform Sentiment Prediction
        emit_progress(task_id, 80, "Predicting sentiments...")
        data = predict_sentiment(data, model)
        print(data.head())
        print(data.columns)

        save_analysis_data(data, task_id)

        # Step 9: Create Dashboard
        emit_progress(task_id, 90, "Creating dashboard...")
        dashboard_paths = create_dashboard(data, task_id)

        # Store task results for rendering later
        task_results[task_id] = {
            "restaurant_name": restaurant_name,
            "analysis_timestamp":  task_id.split('--')[-1],
            "raw_data": data.head().to_html(classes="table table-striped", index=False),
            "preprocessed_data": data.head().to_html(classes="table table-striped", index=False),
            "features": data.head().to_html(classes="table table-striped", index=False),
            "sentiment_labels": data[["cleaned_comment", "sentiment_label"]].head().to_html(classes="table table-striped", index=False),
            "predictions": data[["cleaned_comment", "predicted_sentiment"]].head().to_html(classes="table table-striped", index=False),
            **dashboard_paths
        }

        # Notify completion
        emit_progress(task_id, 100, "Analysis complete!")
    except Exception as e:
        emit_progress(task_id, 100, f"Error: {str(e)}")

@api_blueprint.route("/result/<task_id>", methods=["GET"])
def result(task_id):
    # Retrieve task results
    result_data = task_results.get(task_id)
    if not result_data:
        return render_template("error.html", message="Task results not found!")
    return render_template("results.html", **result_data)
