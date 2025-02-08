from flask import Flask, render_template, request
import pickle
import pandas as pd

dt = pickle.load(open("DecisionTreeClassifier.pkl", "rb"))
gnb = pickle.load(open("Naive Bayes.pkl", "rb"))
knn = pickle.load(open("KNN.pkl", "rb"))
rf = pickle.load(open("RandomForestClassifier.pkl", "rb"))

models = {
    "decision_tree_classifier": dt,
    "random_forest_classifier": rf,
    "knn": knn,
    "naive_bayes": gnb,
}

app = Flask(__name__)


@app.route("/")
def Home():
    return render_template("index.html", models=models)


@app.route("/predict", methods=["POST"])
def predict():
    data = request.form

    try:
        sl = float(data["sl"])
        sw = float(data["sw"])
        pl = float(data["pl"])
        pw = float(data["pw"])
        model_name = data["model"]
        
        if model_name not in models:
            return (
                f"Error: Model '{model_name}' not found. Choose from {list(models.keys())}",
                400,
            )

        feature_names = [
            "sepal length (cm)",
            "sepal width (cm)",
            "petal length (cm)",
            "petal width (cm)",
        ]
        X_new = pd.DataFrame([[sl, sw, pl, pw]], columns=feature_names)

        model = models[model_name]
        prediction = (
            "Setosa"
            if model.predict(X_new)[0] == 0
            else "Versicolor" if model.predict(X_new)[0] == 1 else "Virginica"
        )
        return render_template('result.html',result=prediction)

    except ValueError:
        return "Error: Please enter valid numerical values for all fields.", 400


if __name__ == "__main__":
    app.run(debug=True)
