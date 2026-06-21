import pandas as pd
import joblib
import os
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def train_model():
    """Train the model from dataset.csv"""
    
    if not os.path.exists("dataset.csv"):
        print("⚠️ WARNING: dataset.csv not found. Creating sample dataset...")
        sample_data = {
            "education_level": [3, 3, 2, 2, 1, 0, 3, 2, 1, 2, 3, 2, 1, 3, 2],
            "years_experience": [16, 10, 5, 2, 1, 0, 8, 6, 3, 10, 4, 1, 5, 12, 8],
            "training_hours": [40, 120, 8, 0, 0, 0, 16, 24, 0, 0, 0, 0, 0, 72, 48],  # Changed to hours
            "eligibility": [1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1],
            "label": [1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1]
        }
        df = pd.DataFrame(sample_data)
        df.to_csv("dataset.csv", index=False)
        print("✅ Sample dataset created!")
    else:
        df = pd.read_csv("dataset.csv")
    
    feature_columns = ["education_level", "years_experience", "training_hours", "eligibility"]
    
    missing_cols = [col for col in feature_columns if col not in df.columns]
    if missing_cols:
        print(f"❌ Missing columns in dataset: {missing_cols}")
        return
    
    X = df[feature_columns]
    y = df["label"]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    joblib.dump(model, "model.pkl")
    
    print(f"✅ Model trained successfully! Accuracy: {accuracy:.2%}")
    print(f"📁 Model saved as 'model.pkl'")

if __name__ == "__main__":
    train_model()