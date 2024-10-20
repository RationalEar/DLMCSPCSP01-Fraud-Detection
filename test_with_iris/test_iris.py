import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
import joblib
from sympy.stats.rv import probability

MODEL = 'trained_models/random_forest_model.pkl'

if not os.path.exists(MODEL):
    print("Model does not exist. Loading data and training.")
    
    # Load the Iris dataset
    iris = load_iris()
    
    X = iris.data
    y = iris.target
    
    # Create and train a random forest classifier
    clf = RandomForestClassifier()
    clf.fit(X, y)
    
    # Save the model
    if not os.path.exists("trained_models"):
        os.makedirs("trained_models")
    joblib.dump(clf, MODEL)
else:
    print("Model already exists. Loading model.")

# Load the saved model
loaded_model = joblib.load(MODEL)

# Make predictions on new data
new_data = [[5.1, 3.5, 1.4, 0.2], [7.7, 3.8, 6.7, 2.2], [6.3, 3.3, 4.7, 1.6], [6.4, 3.2, 4.5, 1.5], [6.9, 3.1, 5.4, 2.1]]
predictions = loaded_model.predict(new_data)
probabilities = loaded_model.predict_proba(new_data)
print(predictions)
print(probabilities)