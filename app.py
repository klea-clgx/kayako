from flask import Flask, request, render_template
import joblib

app = Flask(__name__)

# Load the trained models
entity_vectorizer = joblib.load("entity_vectorizer.pkl")
entity_classifier = joblib.load("entity_classification_model.pkl")
category_vectorizer = joblib.load("category_vectorizer.pkl")
category_encoder = joblib.load("category_encoder.pkl")
category_model = joblib.load("category_classification_model.pkl")

# Function to predict entities and category
def predict_entities(text):
    sample_text_vectorized = entity_vectorizer.transform([text])
    predicted_labels = entity_classifier.predict(sample_text_vectorized)
    all_entity_labels = entity_classifier.classes_
    predicted_entities = [label for label, predicted in zip(all_entity_labels, predicted_labels[0]) if predicted == 1]
    if not predicted_entities:
        predicted_entities = ["None"]
    sample_text_category_vectorized = category_vectorizer.transform([text])
    predicted_category_encoded = category_model.predict(sample_text_category_vectorized)
    predicted_category_encoded = predicted_category_encoded.reshape(-1, 1)
    predicted_category = category_encoder.inverse_transform(predicted_category_encoded.ravel())  # Reshape using ravel()
    return predicted_entities, predicted_category[0]



@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        data = request.form["text"]
        entities, category = predict_entities(data)
        return render_template("index.html", entities=entities, category=category)
    else:
        return render_template("index.html", entities=None, category=None)


if __name__ == "__main__":
    app.run(debug=True)
