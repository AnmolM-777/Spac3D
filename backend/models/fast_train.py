import os
import pickle
import json
import numpy as np
from pathlib import Path

from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

STYLES = ["Minimalist", "Modern", "Bohemian", "Industrial", "Scandinavian", "Traditional"]
PCA_COMPONENTS = 32
KNN_K = 3
SAMPLES_PER_STYLE = 10
EMBEDDING_DIM = 2048

MODELS_DIR = Path(__file__).parent
PCA_PATH = MODELS_DIR / "pca.pkl"
KNN_PATH = MODELS_DIR / "knn.pkl"
SVM_PATH = MODELS_DIR / "svm.pkl"
FURNITURE_PATH = MODELS_DIR.parent / "data" / "furniture.json"

def fast_train():
    print("Generating dummy ML models for SPAC3D workaround...")
    
    # 1. Create synthetic data
    X = []
    y = []
    
    rng = np.random.RandomState(42)
    
    for i, style in enumerate(STYLES):
        # Create a "cluster" for each style
        center = rng.randn(EMBEDDING_DIM)
        samples = center + rng.normal(0, 0.5, (SAMPLES_PER_STYLE, EMBEDDING_DIM))
        X.append(samples)
        y.extend([i] * SAMPLES_PER_STYLE)
    
    X = np.vstack(X)
    y = np.array(y)
    
    # 2. Fit PCA
    print(f"Fitting PCA (2048 -> {PCA_COMPONENTS})...")
    pca = PCA(n_components=PCA_COMPONENTS, random_state=42)
    X_reduced = pca.fit_transform(X)
    
    # 3. Fit kNN
    print("Fitting kNN...")
    knn_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier(n_neighbors=KNN_K))
    ])
    knn_pipeline.fit(X_reduced, y)
    
    # 4. Fit SVM
    print("Fitting SVM...")
    svm_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel="rbf", probability=True, random_state=42))
    ])
    svm_pipeline.fit(X_reduced, y)
    
    # 5. Save models
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with open(PCA_PATH, "wb") as f: pickle.dump(pca, f)
    with open(KNN_PATH, "wb") as f: pickle.dump(knn_pipeline, f)
    with open(SVM_PATH, "wb") as f: pickle.dump(svm_pipeline, f)
    print(f"Models saved to {MODELS_DIR}")
    
    # 6. Update furniture.json
    print("Updating furniture.json with dummy feature vectors...")
    with open(FURNITURE_PATH, "r") as f:
        furniture = json.load(f)
        
    style_to_idx = {s.lower(): i for i, s in enumerate(STYLES)}
    
    for item in furniture:
        idx = style_to_idx.get(item["style"].lower(), 0)
        # Random vector around the "style center" in reduced space
        style_center = X_reduced[y == idx].mean(axis=0)
        vec = style_center + rng.normal(0, 0.1, PCA_COMPONENTS)
        item["feature_vector"] = [round(float(v), 6) for v in vec]
        
    with open(FURNITURE_PATH, "w") as f:
        json.dump(furniture, f, indent=2)
    print("furniture.json updated.")
    
    print("\nFast training complete! All models are ready for uvicorn.")

if __name__ == "__main__":
    fast_train()
