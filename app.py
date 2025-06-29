from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load model and scaler
model = pickle.load(open("knn_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    # Get input values from form
    features = [
        float(request.form["f1"]),
        float(request.form["f2"]),
        float(request.form["f3"]),
        float(request.form["f4"]),
        float(request.form["f5"]),
        float(request.form["f6"]),
        float(request.form["f7"]),
        float(request.form["f8"]),
    ]
    
    # Scale and predict
    scaled_features = scaler.transform([features])
    prediction = model.predict(scaled_features)[0]
    result = "Diabetic" if prediction == 1 else "Non-Diabetic"
    
    return render_template("result.html", prediction=result)

if __name__ == "__main__":
    app.run(debug=False)
