# =====================================================
# 1. IMPORT LIBRARIES
# =====================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# =====================================================
# 2. DATASET
# =====================================================
data = []

# =====================================================
# 3. CREATE DATAFRAME & FEATURES
# =====================================================

df = pd.DataFrame(data, columns=["Length", "Breadth", "Label"])
df["Ratio"] = df["Length"] / df["Breadth"]

X = df[["Length", "Breadth", "Ratio"]]
y = df["Label"]

# =====================================================
# 4. DATA VISUALIZATION (FEATURE SPACE)
# =====================================================

plt.figure()
plt.scatter(df[df.Label == 0]["Length"], df[df.Label == 0]["Breadth"], label="Bulath")
plt.scatter(df[df.Label == 1]["Length"], df[df.Label == 1]["Breadth"], label="Gambiris")
plt.xlabel("Length")
plt.ylabel("Breadth")
plt.title("Leaf Feature Distribution")
plt.legend()
plt.show()

# =====================================================
# 5. TRAIN-TEST SPLIT
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =====================================================
# 6. FEATURE SCALING
# =====================================================

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =====================================================
# 7. SVM MODEL
# =====================================================

svm_model = SVC(kernel="rbf", C=1.0, gamma="scale", probability=True)
svm_model.fit(X_train_scaled, y_train)

y_pred_svm = svm_model.predict(X_test_scaled)
svm_probs = svm_model.predict_proba(X_test_scaled)[:, 1]

print("===== SVM RESULTS =====")
print("Accuracy:", accuracy_score(y_test, y_pred_svm))
print(confusion_matrix(y_test, y_pred_svm))
print(classification_report(y_test, y_pred_svm))

# --- SVM Confusion Matrix Plot ---
cm = confusion_matrix(y_test, y_pred_svm)
plt.figure()
plt.imshow(cm)
plt.title("SVM Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.colorbar()
plt.show()

# --- SVM ROC Curve ---
fpr, tpr, _ = roc_curve(y_test, svm_probs)
plt.figure()
plt.plot(fpr, tpr, label="SVM ROC (AUC = %.2f)" % auc(fpr, tpr))
plt.plot([0, 1], [0, 1])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("SVM ROC Curve")
plt.legend()
plt.show()

# =====================================================
# 8. DECISION TREE MODEL
# =====================================================

dt_model = DecisionTreeClassifier(max_depth=5, random_state=42)
dt_model.fit(X_train, y_train)

y_pred_dt = dt_model.predict(X_test)

print("\n===== DECISION TREE RESULTS =====")
print("Accuracy:", accuracy_score(y_test, y_pred_dt))
print(confusion_matrix(y_test, y_pred_dt))
print(classification_report(y_test, y_pred_dt))

# --- Decision Tree Confusion Matrix ---
cm_dt = confusion_matrix(y_test, y_pred_dt)
plt.figure()
plt.imshow(cm_dt)
plt.title("Decision Tree Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.colorbar()
plt.show()

# =====================================================
# 9. NEURAL NETWORK (ANN)
# =====================================================

ann_model = Sequential([
    Dense(16, activation="relu", input_shape=(X_train_scaled.shape[1],)),
    Dense(8, activation="relu"),
    Dense(1, activation="sigmoid")
])

ann_model.compile(
    optimizer=Adam(learning_rate=0.01),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

history = ann_model.fit(
    X_train_scaled,
    y_train,
    epochs=50,
    batch_size=16,
    validation_split=0.1,
    verbose=1
)

# =====================================================
# 10. ANN TRAINING GRAPHS
# =====================================================

plt.figure()
plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.title("ANN Accuracy vs Epochs")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

plt.figure()
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("ANN Loss vs Epochs")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

# =====================================================
# 11. ANN EVALUATION
# =====================================================

ann_probs = ann_model.predict(X_test_scaled).flatten()
ann_preds = (ann_probs >= 0.5).astype(int)

print("\n===== ANN RESULTS =====")
print("Accuracy:", accuracy_score(y_test, ann_preds))
print(confusion_matrix(y_test, ann_preds))
print(classification_report(y_test, ann_preds))

# --- ANN Confusion Matrix ---
cm_ann = confusion_matrix(y_test, ann_preds)
plt.figure()
plt.imshow(cm_ann)
plt.title("ANN Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.colorbar()
plt.show()

# --- ANN ROC Curve ---
fpr_ann, tpr_ann, _ = roc_curve(y_test, ann_probs)
plt.figure()
plt.plot(fpr_ann, tpr_ann, label="ANN ROC (AUC = %.2f)" % auc(fpr_ann, tpr_ann))
plt.plot([0, 1], [0, 1])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ANN ROC Curve")
plt.legend()
plt.show()

# =====================================================
# 12. NEW LEAF PREDICTION
# =====================================================

new_leaf = np.array([[468, 320, 468/320]])
new_leaf_scaled = scaler.transform(new_leaf)

print("\n--- NEW LEAF PREDICTION ---")

print("SVM:", "Gambiris" if svm_model.predict(new_leaf_scaled)[0] == 1 else "Bulath")
print("Decision Tree:", "Gambiris" if dt_model.predict(new_leaf)[0] == 1 else "Bulath")
print("ANN:", "Gambiris" if ann_model.predict(new_leaf_scaled)[0][0] >= 0.5 else "Bulath")
