# =====================================================
# 1. IMPORT LIBRARIES
# =====================================================

import numpy as np
import pandas as pd
import joblib

from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# =====================================================
# 2. DATASET
# Label: 0 = Bulath, 1 = Gambiris
# =====================================================
df = pd.read_csv("leaf_dataset.csv")
df["Ratio"] = df["Length"] / df["Breadth"]

X = df[["Length", "Breadth", "Ratio"]]
y = df["Label"]


# =====================================================
# 3. CREATE DATAFRAME & FEATURES
# =====================================================

df = pd.DataFrame(data, columns=["Length", "Breadth", "Label"])
df["Ratio"] = df["Length"] / df["Breadth"]

X = df[["Length", "Breadth", "Ratio"]]
y = df["Label"]

# =====================================================
# 4. TRAIN-TEST SPLIT
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =====================================================
# 5. FEATURE SCALING
# =====================================================

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =====================================================
# 6. SVM MODEL
# =====================================================

svm_model = SVC(kernel="rbf", C=1.0, gamma="scale", probability=True)
svm_model.fit(X_train_scaled, y_train)

y_pred_svm = svm_model.predict(X_test_scaled)

print("===== SVM RESULTS =====")
print("Accuracy:", accuracy_score(y_test, y_pred_svm))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_svm))
print("Classification Report:\n", classification_report(y_test, y_pred_svm))

# =====================================================
# 7. DECISION TREE MODEL
# =====================================================

dt_model = DecisionTreeClassifier(
    criterion="gini",
    max_depth=5,
    random_state=42
)

dt_model.fit(X_train, y_train)

y_pred_dt = dt_model.predict(X_test)

print("\n===== DECISION TREE RESULTS =====")
print("Accuracy:", accuracy_score(y_test, y_pred_dt))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_dt))
print("Classification Report:\n", classification_report(y_test, y_pred_dt))

# =====================================================
# 8. NEURAL NETWORK (ANN)
# =====================================================

ann_model = Sequential([
    Dense(16, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    Dense(8, activation='relu'),
    Dense(1, activation='sigmoid')
])

ann_model.compile(
    optimizer=Adam(learning_rate=0.01),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

ann_model.fit(
    X_train_scaled,
    y_train,
    epochs=50,
    batch_size=16,
    validation_split=0.1,
    verbose=1
)

# =====================================================
# 9. ANN EVALUATION
# =====================================================

ann_probs = ann_model.predict(X_test_scaled)
ann_preds = (ann_probs >= 0.5).astype(int).flatten()

print("\n===== ANN RESULTS =====")
print("Accuracy:", accuracy_score(y_test, ann_preds))
print("Confusion Matrix:\n", confusion_matrix(y_test, ann_preds))
print("Classification Report:\n", classification_report(y_test, ann_preds))

# =====================================================
# 10. PREDICT NEW LEAF (SVM + DT + ANN)
# =====================================================

new_leaf = np.array([[14.3, 10.4, 14.3 / 10.4]])

# --- SVM Prediction ---
new_leaf_scaled = scaler.transform(new_leaf)
svm_prob = svm_model.predict_proba(new_leaf_scaled)[0][1]
print("\nSVM Probability of Gambiris:", svm_prob)
print("SVM Prediction:", "Gambiris" if svm_prob >= 0.5 else "Bulath")

# --- Decision Tree Prediction ---
dt_pred = dt_model.predict(new_leaf)[0]
print("\nDecision Tree Prediction:", "Gambiris" if dt_pred == 1 else "Bulath")

# --- ANN Prediction ---
ann_prob = ann_model.predict(new_leaf_scaled)[0][0]
print("\nANN Probability of Gambiris:", ann_prob)
print("ANN Prediction:", "Gambiris" if ann_prob >= 0.5 else "Bulath")


joblib.dump(ann_model, "leaf_model.pkl")
joblib.dump(scaler, "scaler.pkl")

