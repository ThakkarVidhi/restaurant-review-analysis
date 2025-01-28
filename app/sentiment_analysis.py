import os
import re
import joblib
import time
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import folium
import nltk
from collections import Counter
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from wordcloud import WordCloud
from datetime import datetime
from config import Config
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from nltk import pos_tag
from folium.plugins import MarkerCluster
from geopy.geocoders import Nominatim
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# Load Trained Model
def load_model(model_path):
    """Loads the trained model from a .pkl file."""
    with open(model_path, 'rb') as file:
        model = joblib.load(file)
    return model

# Preprocess New Data
def clean_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove special characters
    text = text.lower()  # Convert to lowercase
    return text

def preprocess_data(data):
    data.dropna(subset=['comment'], inplace=True)  # Remove rows with missing comments
    data['cleaned_comment'] = data['comment'].apply(clean_text)  # Clean comments
    data['comment_length'] = data['cleaned_comment'].apply(len)  # Add comment length

    # Drop or fill missing values in rating columns
    rating_columns = ['overall_rating', 'food_rating', 'service_rating', 'ambience_rating']
    for col in rating_columns:
        data[col].fillna(data[col].median(), inplace=True)

    data.reset_index(drop=True, inplace=True)
    return data

# Advanced Feature Extraction
def extract_advanced_features(text, city):
    analyzer = SentimentIntensityAnalyzer()

    sentiment = analyzer.polarity_scores(text)
    tokens = text.split()
    
    # Count words by polarity
    pos_count = sum(1 for word in tokens if analyzer.polarity_scores(word)['compound'] > 0.05)
    neu_count = sum(1 for word in tokens if -0.05 <= analyzer.polarity_scores(word)['compound'] <= 0.05)
    neg_count = sum(1 for word in tokens if analyzer.polarity_scores(word)['compound'] < -0.05)
    
    # N-gram extraction (for unigrams, bigrams, trigrams) with polarity analysis
    vectorizer = CountVectorizer(ngram_range=(1, 3))
    n_grams_vector = vectorizer.fit_transform([text])
    n_grams = {ngram: n_grams_vector.toarray().sum(axis=0)[idx] for idx, ngram in enumerate(vectorizer.get_feature_names_out())}

    # Filter n-grams based on polarity
    unigrams = {k: v for k, v in n_grams.items() if len(k.split()) == 1}
    bigrams = {k: v for k, v in n_grams.items() if len(k.split()) == 2}
    trigrams = {k: v for k, v in n_grams.items() if len(k.split()) == 3}
    
    features = {
        'char_count': len(text),
        'word_count': len(tokens),
        'avg_word_length': np.mean([len(word) for word in tokens]) if tokens else 0,
        'sentiment_compound': sentiment['compound'],
        'sentiment_pos': sentiment['pos'],
        'sentiment_neu': sentiment['neu'],
        'sentiment_neg': sentiment['neg'],
        'positive_word_count': pos_count,
        'neutral_word_count': neu_count,
        'negative_word_count': neg_count,
        'unigrams': unigrams,
        'bigrams': bigrams,
        'trigrams': trigrams
    }
    return pd.Series(features)

# Assign Sentiment Labels
def assign_sentiment_label(row):
    if row['sentiment_compound'] >= 0.05:
        return 'positive'
    elif row['sentiment_compound'] <= -0.05:
        return 'negative'
    else:
        return 'neutral'

# Predict Sentiment using Pretrained Model
def predict_sentiment(data, model):
    """Predicts sentiment using the pretrained model."""
    try:
        rating_columns = ['overall_rating', 'food_rating', 'service_rating', 'ambience_rating']

        # One-hot encode the 'location' column
        location_encoded = pd.get_dummies(data['city'], prefix='location')

        # Combine numerical and encoded categorical features
        X = pd.concat([data[rating_columns], 
                    data[['char_count', 'word_count', 'avg_word_length', 'sentiment_compound', 
                            'sentiment_pos', 'sentiment_neu', 'sentiment_neg', 
                            'positive_word_count', 'neutral_word_count', 'negative_word_count']],
                    location_encoded], axis=1)

        predictions = model.predict(X)

        data['predicted_sentiment'] = predictions
        data['sentiment_label'] = data.apply(assign_sentiment_label, axis=1)
    except Exception as e:
        raise RuntimeError(f"Prediction failed: {e}")
    return data

# Generate Word Clouds
def generate_wordcloud(data, sentiment):
    """Generates a word cloud for the specified sentiment."""
    text = ' '.join(data[data['predicted_sentiment'] == sentiment]['cleaned_comment'])
    if not text:
        st.warning(f"No comments for {sentiment} sentiment.")
        return
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f"Word Cloud for {sentiment.capitalize()} Sentiment")
    st.pyplot()

# Top N Grams
def plot_top_ngrams(n_grams_dict, title):
    """Plots the top n-grams."""
    top_ngrams = Counter(n_grams_dict).most_common(10)
    ngrams, frequencies = zip(*top_ngrams)
    plt.barh(ngrams, frequencies, color='skyblue')
    plt.xlabel("Frequency")
    plt.title(title)
    plt.gca().invert_yaxis()
    st.pyplot()

# Create Streamlit Dashboard
def create_dashboard(data, task_id):
    try:
        output_dir = os.path.join(Config.DASHBOARD_DATA_DIR, task_id)
        os.makedirs(output_dir, exist_ok=True)

        # Paths for saving files and generating URLs
        rel_output_dir = os.path.relpath(output_dir, Config.STATIC_DIR)

        # stop words list in english
        stop_words = set(stopwords.words('english'))

        # Filter function to remove stop words
        def filter_stopwords(text):
            words = text.split()
            return [word for word in words if word.lower() not in stop_words]

        # Function to update the DataFrame with proper keyword and frequency counts
        def generate_keyword_df(filtered_words):
            keyword_counter = Counter()
            for keyword, freq in Counter(filtered_words).items():
                keyword_counter[keyword] += freq
            keyword_df = pd.DataFrame(keyword_counter.items(), columns=['Keyword', 'Frequency'])
            return keyword_df

        # Function to get latitude and longitude using geopy
        def get_lat_lon(city_name):
            try:
                location = geolocator.geocode(city_name, timeout=10) 
                if location:
                    return location.latitude, location.longitude
                else:
                    return None, None
            except Exception as e:
                print(f"Error geocoding {city_name}: {e}")
                return None, None

        # Insights on sentiment labels
        insights = []
        for sentiment in ['positive', 'neutral', 'negative']:
            count = len(data[data['sentiment_label'] == sentiment])
            percentage = count / len(data) * 100
            insights.append(f"{sentiment.capitalize()} Reviews: {count} ({percentage:.2f}%)")

        # Polarity Distribution Plot
        polarity_dist_path = os.path.join(output_dir, "polarity_distribution.png")
        plt.figure(figsize=(8, 6))
        sns.histplot(data['sentiment_compound'], kde=True, bins=30)
        plt.title("Polarity Score Distribution")
        plt.xlabel("Compound Score")
        plt.ylabel("Frequency")
        plt.savefig(polarity_dist_path, bbox_inches="tight")
        plt.close()

        # Word Clouds for Sentiments
        wordcloud_paths = {}
        for sentiment in ['positive', 'neutral', 'negative']:
            wordcloud = WordCloud(width=800, height=400, background_color="white").generate(
                " ".join(data[data['sentiment_label'] == sentiment]['cleaned_comment'])
            )
            wordcloud_path = os.path.join(output_dir, f"wordcloud_{sentiment}.png")
            wordcloud.to_file(wordcloud_path)
            wordcloud_paths[sentiment] = f"/static/{rel_output_dir}/wordcloud_{sentiment}.png"

        # Review Length and Sentiment Correlation
        length_sentiment_path = os.path.join(output_dir, "length_sentiment_correlation.png")
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x='char_count', y='sentiment_compound', data=data)
        plt.title("Review Length vs Sentiment")
        plt.xlabel("Character Count")
        plt.ylabel("Sentiment Compound Score")
        plt.savefig(length_sentiment_path, bbox_inches="tight")
        plt.close()

        # Sample data for sentiment (replace with your actual data)
        city_sentiment = data.groupby('city')['sentiment_label'].value_counts().unstack(fill_value=0)

        # Geocoder initialization
        geolocator = Nominatim(user_agent="city_sentiment_map")

        # Create the map centered at a default location (e.g., global center) initially
        city_map = folium.Map(location=[20.0, 0.0], zoom_start=2)
        marker_cluster = MarkerCluster().add_to(city_map)

        # Iterate over cities and create markers
        for city_name in city_sentiment.index:
            latitude, longitude = get_lat_lon(city_name)
            if latitude and longitude:
                sentiment_counts = city_sentiment.loc[city_name].to_dict()
                popup_text = f"<b>{city_name}</b><br>" + "<br>".join([f"{k}: {v}" for k, v in sentiment_counts.items()])
                folium.Marker(location=[latitude, longitude], popup=popup_text).add_to(marker_cluster)
            else:
                print(f"Could not find coordinates for {city_name}")
            
            # Sleep for 1 second between requests to avoid rate-limiting
            time.sleep(1)

        # Save the map as an HTML file
        map_path = os.path.join(output_dir, 'city_sentiment_map.html')
        city_map.save(map_path)

        # Inject a "Back to Results" button
        back_button_html = f"""
        <div style="position:fixed;top:10px;left:10px;z-index:999;">
            <a href="/result/{task_id}" class="btn btn-primary">Back to Results</a>
        </div>
        """
        with open(map_path, 'r') as file:
            map_html = file.read()

        map_html_with_button = back_button_html + map_html

        with open(map_path, 'w') as file:
            file.write(map_html_with_button)

        # Positive Aspects Analysis
        positive_aspects = (
            data[data['sentiment_pos'] > 0]
            .nlargest(20, 'positive_word_count')['cleaned_comment']
            .str.cat(sep=' ')
        )
        positive_filtered_words = filter_stopwords(positive_aspects)
        positive_keywords = pd.Series(positive_filtered_words).value_counts().head(20)
        print("positive_keywords: ", positive_keywords)

        plt.figure(figsize=(10, 6))
        positive_keywords.plot(kind="bar", color="green")
        plt.title("Top Positive Keywords")
        plt.ylabel("Frequency")
        plt.xlabel("Words")
        positive_aspects_path = os.path.join(output_dir, "positive_aspects.png")
        plt.savefig(positive_aspects_path, bbox_inches="tight")
        plt.close()

        # Negative Aspects Analysis
        negative_aspects = (
            data[data['sentiment_neg'] > 0]
            .nlargest(20, 'negative_word_count')['cleaned_comment']
            .str.cat(sep=' ')
        )
        negative_filtered_words = filter_stopwords(negative_aspects)
        negative_keywords = pd.Series(negative_filtered_words).value_counts().head(20)
        print("negative_keywords: ", negative_keywords)

        plt.figure(figsize=(10, 6))
        negative_keywords.plot(kind="bar", color="red")
        plt.title("Top Negative Keywords")
        plt.ylabel("Frequency")
        plt.xlabel("Words")
        negative_aspects_path = os.path.join(output_dir, "negative_aspects.png")
        plt.savefig(negative_aspects_path, bbox_inches="tight")
        plt.close()

        # Calculate average ratings
        avg_service_rating = data['service_rating'].mean()
        avg_food_rating = data['food_rating'].mean()
        avg_ambience_rating = data['ambience_rating'].mean()

        # Initialize lists for aggregated suggestions and sentiment analysis plots
        aggregated_suggestions = []
        category_sentiment_paths = {}

        # Categories to compare: Food, Service, Ambience
        categories = ['food_rating', 'service_rating', 'ambience_rating']
        category_comparisons = {
            "service": avg_service_rating,
            "food": avg_food_rating,
            "ambience": avg_ambience_rating
        }

        # Generate sentiment plots for each category and store the plot paths
        for category in categories:
            plt.figure(figsize=(8, 6))
            sns.boxplot(x=category, y='sentiment_compound', data=data)
            plt.title(f"Sentiment vs {category.replace('_', ' ').capitalize()}")
            plt.xlabel(category.replace('_', ' ').capitalize())
            plt.ylabel("Sentiment Compound Score")
            category_path = os.path.join(output_dir, f"sentiment_by_{category}.png")
            plt.savefig(category_path, bbox_inches="tight")
            plt.close()
            category_sentiment_paths[category] = f"/static/{rel_output_dir}/sentiment_by_{category}.png"

        # Create the category comparison plot (Radar/Spider Chart)
        categories_array = ['Food Rating', 'Service Rating', 'Ambience Rating']
        values = [avg_food_rating, avg_service_rating, avg_ambience_rating]

        # Number of variables
        num_vars = len(categories_array)

        # Compute angle for each axis
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()

        # Close the plot loop
        values += values[:1]
        angles += angles[:1]

        # Create the Radar plot
        fig, ax = plt.subplots(figsize=(6, 6), dpi=100, subplot_kw=dict(polar=True))
        ax.fill(angles, values, color='orange', alpha=0.25)
        ax.plot(angles, values, color='orange', linewidth=2)
        ax.set_yticklabels([])  # Hide the radial ticks

        # Add category labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories_array)

        # Save the figure
        category_comparison_path = os.path.join(output_dir, "category_comparison_radar.png")
        plt.savefig(category_comparison_path, bbox_inches="tight")
        plt.close()

        category_comparison_image_path = f"/static/{rel_output_dir}/category_comparison_radar.png"

        # Create the improvements dictionary
        improvements = {
            "avg_ratings": {
                "service": round(avg_service_rating, 2),
                "food": round(avg_food_rating, 2),
                "ambience": round(avg_ambience_rating, 2)
            },
            "aggregated_suggestions": aggregated_suggestions,
            "category_sentiments": category_sentiment_paths,
            "category_comparison": category_comparison_image_path
        }

        # Analysis
        analysis = {}

        # Highly Positive Reviews (Engagement)
        satisfied_customers = [
            {
                "user_name": review['user_name'],
                "city": review['city'],
                "comment": review['comment'],
                "rating": review['overall_rating']
            }
            for _, review in data.iterrows()
            if review.get('overall_rating', 0) >= 4.5
        ]

        # Least satisfied customers
        least_satisfied_customers = [
            {
                "user_name": review['user_name'],
                "city": review['city'],
                "comment": review['comment'],
                "rating": review['overall_rating']
            }
            for _, review in data.iterrows()
            if review.get('overall_rating', 5) <= 2.0
        ]

        # Limit to top 5 for analysis
        analysis['loyal_customers'] = satisfied_customers[:5]
        analysis['least_satisfied_customers'] = least_satisfied_customers[:5]

        # Sentiment trends: Aggregate sentiment values
        sentiment_pos = data['sentiment_pos'].fillna(0).sum()  # Sum of positive sentiment
        sentiment_neg = data['sentiment_neg'].fillna(0).sum()  # Sum of negative sentiment

        # Pie chart for sentiment trends
        plt.pie(
            [sentiment_pos, sentiment_neg],
            labels=['Positive Sentiment', 'Negative Sentiment'],
            autopct='%1.1f%%',
            colors=['green', 'red']
        )
        plt.title('Sentiment Trends')
        sentiment_chart_path = os.path.join(output_dir, 'sentiment_trends.png')
        plt.savefig(sentiment_chart_path)
        plt.close()

        # Return paths to the new visualizations
        return {
            "insights": insights,
            "polarity_distribution": f"/static/{rel_output_dir}/polarity_distribution.png",
            "wordclouds": wordcloud_paths,
            "length_sentiment": f"/static/{rel_output_dir}/length_sentiment_correlation.png",
            "city_sentiment_map_url": f"/static/{rel_output_dir}/city_sentiment_map.html",
            "positive_aspects": f"/static/{rel_output_dir}/positive_aspects.png",
            "negative_aspects": f"/static/{rel_output_dir}/negative_aspects.png",
            "improvements": improvements,
            "sentiment_trends": f"/static/{rel_output_dir}/sentiment_trends.png",
            "analysis": analysis
        }
    except Exception as e:
        print(f"Error: {e}")
        return None

def save_analysis_data(data, task_id):
    try:
        # Construct the file name & path
        csv_file_name = f"{task_id}.csv"
        csv_file_path = os.path.join(Config.ANALYSIS_DATA_DIR, csv_file_name)

        # Ensure the analysis data directory exists
        os.makedirs(Config.ANALYSIS_DATA_DIR, exist_ok=True)

        # Save analysis reviews to CSV
        df = pd.DataFrame(data)
        df.to_csv(csv_file_path, index=False)
        print(f"Data saved to {csv_file_path}")   
    except Exception as e:
        print(f"Error saving file: {e}")
        return None