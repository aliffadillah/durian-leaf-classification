import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import joblib
import json
import numpy as np

# Load data
print("Loading GLCM features data...")
df = pd.read_csv("glcm_features.csv")
print(f"Dataset shape: {df.shape}")
print(f"Classes: {df['label'].unique()}")
print(f"Class distribution:\n{df['label'].value_counts()}")

# Prepare features and labels
X = df.drop("label", axis=1)
y = df["label"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTraining samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")

# Feature scaling (important for KNN)
print("\nApplying feature scaling...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Hyperparameter tuning for KNN
print("\nPerforming hyperparameter tuning...")
param_grid = {
    'n_neighbors': [3, 5, 7, 9, 11, 13, 15],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan', 'minkowski']
}

knn = KNeighborsClassifier()
grid_search = GridSearchCV(
    knn, 
    param_grid, 
    cv=5, 
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train_scaled, y_train)

print(f"\nBest parameters: {grid_search.best_params_}")
print(f"Best cross-validation score: {grid_search.best_score_:.4f}")

# Train final model with best parameters
best_model = grid_search.best_estimator_
print(f"\nFinal model: {best_model}")

# Make predictions
y_pred = best_model.predict(X_test_scaled)

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)
print(f"\nTest Accuracy: {accuracy:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Calculate feature importance (for KNN, we can use feature variance as proxy)
feature_variance = np.var(X_train_scaled, axis=0)
feature_importance = feature_variance / np.sum(feature_variance)

# Save model and scaler
print("\nSaving model and scaler...")
joblib.dump(best_model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")

# Save model information
model_info = {
    "model_type": "KNeighborsClassifier",
    "classes": sorted(y.unique().tolist()),
    "feature_names": X.columns.tolist(),
    "feature_importance": feature_importance.tolist(),
    "best_params": grid_search.best_params_,
    "cv_score": float(grid_search.best_score_),
    "test_accuracy": float(accuracy),
    "training_samples": len(X_train),
    "test_samples": len(X_test),
    "total_samples": len(df),
    "scaling_applied": True
}

with open("model_info.json", "w") as f:
    json.dump(model_info, f, indent=2)

print(f"\nModel saved as 'model.pkl'")
print(f"Scaler saved as 'scaler.pkl'")
print(f"Model info saved as 'model_info.json'")

# Display final results
print("\n" + "="*50)
print("TRAINING COMPLETED")
print("="*50)
print(f"Model Type: K-Nearest Neighbors")
print(f"Best Parameters: {grid_search.best_params_}")
print(f"Cross-validation Score: {grid_search.best_score_:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")
print(f"Classes: {model_info['classes']}")
print("="*50)
