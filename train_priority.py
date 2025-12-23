import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib

df = pd.read_csv("data/tickets_raw.csv")

# Keep only what we need
df = df[["body", "priority", "language"]].dropna()

# English only
df = df[df["language"] == "en"]

# Remove super-rare priorities (just in case)
df = df[df["priority"].map(df["priority"].value_counts()) >= 20]

print(f"Loaded {len(df)} English tickets for PRIORITY model")

X = df["body"]
y = df["priority"].astype(str)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english", max_features=20000, ngram_range=(1, 2))),
    ("clf", LogisticRegression(max_iter=3000))
])

model.fit(X_train, y_train)

joblib.dump(model, "priority_model.joblib")
print("✅ Priority model trained and saved as priority_model.joblib")
