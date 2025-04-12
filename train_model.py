import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib

# Load dataset
df = pd.read_csv("decor_data.csv")

# Features and target
X = df["Description"]
y = df["Recommendation"]

# Create a simple pipeline
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', LogisticRegression())
])

# Train the model
pipeline.fit(X, y)

# Save the model
joblib.dump(pipeline, "model.joblib")