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

data = [

[18, 9.1, 0],[17.4, 9.5, 0],[17.6, 9.3, 0],[17.6, 9.1, 0],[17.3, 8.1, 0],[17.2, 10.4, 0],
[14.7, 10.9, 0],[16, 10.6, 0],[16.3, 9.2, 0],[17.1, 9.6, 0],[15.7, 11.6, 0],[16.6, 10.4, 0],
[16.5, 10.1, 0],[17.3, 9.5, 0],[15.9, 10.4, 0],[15.2, 11.3, 0],[16.6, 12.1, 0],
[16.5, 10.4, 0],[15.3, 9.8, 0],[16.2, 9, 0],[15.4, 9.8, 0],[15.9, 9.7, 0],
[16.9, 10.1, 0],[17.1, 10.5, 0],[16.4, 10.8, 0],[13.1, 8, 0],[15.3, 11.4, 0],
[17.4, 9.4, 0],[16.6, 10.4, 0],[15.4, 8.6, 0],[16.4, 10.2, 0],[15.3, 9.1, 0],
[16.8, 11.3, 0],[15.1, 10.2, 0],[14.7, 8.7, 0],[13.8, 9.5, 0],[17.1, 9.2, 0],
[14.8, 9, 0],[13.7, 9.9, 0],[14.8, 11.3, 0],[17, 8.5, 0],[16.4, 11.2, 0],
[18.1, 9.4, 0],[15.4, 10.3, 0],[16.4, 10.5, 0],
[18.1, 9.5, 0],[18.3, 10.5, 0],[16.5, 9.6, 0],[16.7, 9, 0],[19.8, 9.6, 0],
[17.1, 10.4, 0],[18, 10.2, 0],[17.4, 9.2, 0],[17.2, 9.1, 0],[17.5, 11, 0],
[17.4, 11.6, 0],[17.2, 10.4, 0],[16.8, 10.1, 0],[17.6, 10.1, 0],[14.5, 11.2, 0],
[16.1, 10.5, 0],[16.5, 11.7, 0],[15.2, 10.3, 0],[15, 9.4, 0],[17.1, 10.5, 0],
[14.7, 12, 0],[16.9, 9.9, 0],[17, 11, 0],[16.8, 9.6, 0],[17.5, 9.7, 0],
[16.4, 9.2, 0],[17.3, 10.2, 0],[15.7, 9, 0],[21.7, 10.5, 0],[15.3, 9.2, 0],
[17.8, 10, 0],[15.3, 9.4, 0],[17.3, 10.4, 0],[15.7, 10.3, 0],[16, 9.6, 0],
[17.3, 11.2, 0],[14.9, 10.6, 0],[15.1, 10.1, 0],[15.8, 11.3, 0],[18.2, 9.5, 0],
[16.5, 9.7, 0],[17, 10.4, 0],[17.5, 9.2, 0],[16.7, 10.5, 0],[14.8, 8.8, 0],
[17.4, 9, 0],[16.9, 10.8, 0],[17.6, 11.7, 0],[15.7, 9.9, 0],[17.3, 11.1, 0],
[17.8, 10.2, 0],

# -------- GAMBIRIS --------
[14.3, 10.4, 1],[13.1, 9.7, 1],[14.4, 9.6, 1],[12, 7.9, 1],[13.5, 8, 1],[12.6, 9.3, 1],
[12.9, 7.8, 1],[8.5, 13.1, 1],[13, 10.7, 1],[12.5, 8.2, 1],[13.5, 10.5, 1],[12.2, 9.2, 1],
[14.3, 9.7, 1],[13.5, 9.9, 1],[13.2, 9.7, 1],[11.6, 9, 1],[13.2, 9.6, 1],[12.4, 8, 1],
[12, 8.9, 1],[12.2, 9, 1],[13.8, 10.2, 1],[12.7, 8.5, 1],[14.1, 9.6, 1],[12.2, 7.7, 1],
[13.1, 8.7, 1],[15.2, 7.8, 1],[14.3, 7.5, 1],[13.2, 8.2, 1],[13.5, 8, 1],[13.4, 8.5, 1],
[13.5, 8.8, 1],[13, 7.5, 1],[14.5, 9.9, 1],[13.2, 7.2, 1],[14.4, 6.8, 1],[14, 7.4, 1],
[13.3, 7.2, 1],[13.1, 6.2, 1],[15, 9.4, 1],[16.6, 8.7, 1],[14.9, 9.3, 1],[13.5, 8, 1],
[12.7, 8.1, 1],[15.2, 8.2, 1],[13.4, 7.3, 1],[11.5, 8.2, 1],[13.5, 7.7, 1],[13.8, 9, 1],
[13.4, 7.6, 1],[13.3, 9, 1],[13.8, 8.6, 1],[11.7, 7.4, 1],[12.2, 8, 1],[11.3, 8.1, 1],
[12.9, 9.1, 1],[13.3, 7.7, 1],[9.7, 5.2, 1],[10.4, 6.2, 1],[12, 7.5, 1],[12.8, 8, 1],
[13, 8.5, 1],[12.3, 8.5, 1],[16, 10.9, 1],[13.2, 9.4, 1],[11.5, 7.3, 1],[13.5, 8, 1],
[15.1, 10.1, 1],[12.4, 7.3, 1],[15.2, 8.7, 1],[11.5, 7.6, 1],[14.9, 9.3, 1],[14.3, 8.5, 1],
[13.6, 8.1, 1],[14, 7.5, 1],[12.9, 8, 1],[13.1, 7.5, 1],[13.6, 8.1, 1],[14.4, 7.5, 1],
[13.8, 7.6, 1],[15.9, 9.2, 1],[12.8, 7.7, 1],[15, 8.4, 1],[14.6, 9.1, 1],[12.6, 7.5, 1],
[14.3, 9.6, 1],[14.7, 9.1, 1],[16.2, 7.8, 1],[14.7, 8.2, 1],[13.3, 7.1, 1],[14, 8.4, 1],
[13.9, 7.8, 1],[15.8, 9.4, 1],[15.1, 8.9, 1],[15.4, 8.4, 1],[11.8, 7, 1],[13.1, 7.2, 1],
[15.3, 8, 1],[14.5, 8.5, 1],[16.3, 9.5, 1],[13.5, 7, 1],[16.3, 9.5, 1],[15.4, 8.6, 1],
[14.4, 8, 1],[14.8, 8, 1],[13.7, 7.7, 1],[14.5, 9.9, 1],[11.7, 8.1, 1],[10.1, 5.5, 1],
[14, 7.1, 1],[15.5, 7, 1],[14.1, 9.3, 1],[11, 7.3, 1],[12.7, 6.3, 1],[14.3, 8.3, 1],
[15, 9, 1],[11.6, 7.3, 1],[9.5, 4.8, 1],[14.8, 8.6, 1],[15.5, 9.5, 1],[16, 10.2, 1],
[15.3, 11.3, 1],[15, 7.3, 1],[12.8, 6.7, 1],[12.8, 8, 1],[14.6, 8, 1],[13.2, 7.7, 1],
[16, 9.3, 1],[16, 8.3, 1],[15, 8.3, 1],[9.6, 4.3, 1],[15.5, 7.6, 1],[14.9, 7, 1],
[14, 7.3, 1],[16, 8.7, 1],[13, 6.6, 1],[15.3, 9.5, 1],[13.5, 8.6, 1],[13, 6.6, 1],
[14.2, 8.6, 1],[16.1, 9.8, 1],[15.6, 10.5, 1],[13.3, 7.8, 1],[13.2, 7.5, 1],[9.7, 5.9, 1],
[16.7, 10.3, 1],[14.1, 9.5, 1],[13.4, 8.7, 1],[13.7, 9.2, 1],[14.2, 7.3, 1],[14.3, 8.7, 1]
]

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
