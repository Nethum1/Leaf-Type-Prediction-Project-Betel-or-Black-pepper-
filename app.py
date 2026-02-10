from flask import Flask, render_template, request
import cv2
import numpy as np
import joblib
import os

# -----------------------------
# FLASK SETUP
# -----------------------------
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"

# Load saved ML model and scaler
model = joblib.load("leaf_model.pkl")   # Your trained SVM / DT / ANN
scaler = joblib.load("scaler.pkl")

# -----------------------------
# FUNCTION: EXTRACT LEAF FEATURES
# -----------------------------
def extract_leaf_features(image_path):
    img = cv2.imread(image_path)

    # Resize for consistent measurement
    img = cv2.resize(img, (500, 500))

    # Convert to HSV to detect green leaf
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Define range for green color (adjust if needed)
    lower = np.array([25, 40, 40])
    upper = np.array([85, 255, 255])

    mask = cv2.inRange(hsv, lower, upper)

    # Morphological operations to remove noise
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return 0, 0, 0  # No leaf detected

    # Largest contour assumed to be leaf
    leaf_contour = max(contours, key=cv2.contourArea)

    # Bounding rectangle
    x, y, w, h = cv2.boundingRect(leaf_contour)

    # Pixel measurements
    length = max(w, h)
    width = min(w, h)
    ratio = length / width if width != 0 else 0

    return length, width, ratio

# -----------------------------
# ROUTES
# -----------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["leaf"]
        if file.filename == "":
            return render_template("index.html", error="No file selected")

        filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(filepath)

        # Extract features
        length, width, ratio = extract_leaf_features(filepath)
        print(f"Length: {length}, Width: {width}, Ratio: {ratio}")

        if length == 0:
            return render_template("index.html", error="Leaf not detected. Try a clearer image.")

        # Scale features
        X = scaler.transform([[length, width, ratio]])

        # Predict variety
        prediction = model.predict(X)[0]
        variety = "Gambiris" if prediction == 1 else "Bulath"

        return render_template(
            "result.html",
            length=round(length, 2),
            width=round(width, 2),
            ratio=round(ratio, 2),
            variety=variety
        )

    return render_template("index.html")

# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False, use_reloader=False)
