import os
import joblib
import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from loguru import logger

# Configuration
MODEL_DIR = "model_store"
MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")

def train():
    logger.info("Starting training pipeline...")
    
    # 1. Load Data (Iris dataset for simplicity)
    # Features: sepal length, sepal width, petal length, petal width
    data = load_iris()
    X, y = data.data, data.target
    
    # 2. Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 3. Train Model (RandomForest simulating a complex model)
    clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    clf.fit(X_train, y_train)
    
    # 4. Evaluate
    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    logger.info(f"Model trained. Accuracy: {acc:.4f}")
    
    # 5. Save Artifact
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
        
    joblib.dump(clf, MODEL_PATH)
    logger.info(f"Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    train()