
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib

# Load real dataset
df = pd.read_csv("data/tickets_raw.csv")

# Keep only rows we need
df = df[["body", "type", "language"]].dropna()

# Use only English tickets (important for accuracy)
df = df[df["language"] == "en"]

print(f"Loaded {len(df)} English tickets")

X = df["body"]
y = df["type"]

# Train / test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Build model
model = Pipeline([
    ("tfidf", TfidfVectorizer(
        stop_words="english",
        max_features=20000,
        ngram_range=(1, 2)
    )),
    ("clf", LogisticRegression(max_iter=3000))
])

# Train
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "ticket_model.joblib")

print("✅ Model trained on real dataset and saved as ticket_model.joblib")
