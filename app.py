from flask import Flask, render_template, request
import json
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder
import numpy as np
app = Flask(__name__)

# Load the data from admin.jsonl
data = []
with open("admin.jsonl", "r", encoding="utf-8") as file:
    for line in file:
        entry = json.loads(line)
        data.append(entry)

# Convert the data into a DataFrame
df = pd.DataFrame(data)

# Load the trained entity classification model and vectorizer
vectorizer = joblib.load("entity_vectorizer_admin.pkl")
entity_classifier = joblib.load("entity_classification_model_admin.pkl")

# Load the trained category classification model and vectorizer
category_vectorizer = joblib.load("category_vectorizer_admin.pkl")
category_model = joblib.load("category_classification_model_admin.pkl")

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        # Get the text input from the user
        sample_text = request.form.get("text_input")

        # Load the trained models and vectorizers
        vectorizer = joblib.load("entity_vectorizer_admin.pkl")
        entity_classifier = joblib.load("entity_classification_model_admin.pkl")

        category_vectorizer = joblib.load("category_vectorizer_admin.pkl")
        category_model = joblib.load("category_classification_model_admin.pkl")

        # Function to get character ranges for each entity
    
        def get_character_ranges(text, entity):
            # Function to get character ranges for each entity
            ranges = []
            start_idx = 0
            words = text.split()
            for i in range(len(words)):
                word = words[i]
                end_idx = start_idx + len(word)
                if word in entity:
                    ranges.append((start_idx, end_idx, word))
                start_idx = end_idx + 1
            return ranges
    
        def get_category_label(encoded_category):
            # Function to get the category label from encoded category
            label = category_encoder.inverse_transform(np.array(encoded_category).reshape(-1, 1))
            return label[0]
        # Use the entity classification model to predict entities in the sample text
        sample_text_vectorized = vectorizer.transform([sample_text])
        predicted_labels = entity_classifier.predict(sample_text_vectorized)
        binarizer = MultiLabelBinarizer()
        predicted_entities = binarizer.inverse_transform(predicted_labels)

        # Use the predicted entities to get character ranges in the sample text
        entities_with_ranges = []
        for entity in predicted_entities[0]:
            ranges = get_character_ranges(sample_text, entity)
            entities_with_ranges.append((entity, ranges))

        # Use the category classification model to predict the email category
        sample_text_category_vectorized = category_vectorizer.transform([sample_text])
        predicted_category_encoded = category_model.predict(sample_text_category_vectorized)
        category_encoder = LabelEncoder()
        category_encoder.classes_ = np.load("category_encoder_classes.npy")
        predicted_category_text = get_category_label(predicted_category_encoded)

        # Pass the results to the result.html template
        return render_template("result.html", sample_text=sample_text, entities_with_ranges=entities_with_ranges, predicted_category_text=predicted_category_text[0])

    return render_template("index.html")



if __name__ == "__main__":
    app.run(debug=True)
