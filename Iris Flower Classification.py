                                            #Iris Flower Classification
#Using Algorithms K-Nearest Neighbors (KNN).
# importing libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score

#loading the dataset 
from sklearn.datasets import load_iris
iris = load_iris()
X = iris.data
y = iris.target

# SPLITTING TEST
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Scaling Features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

k = 3  # You can choose any value for k
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train, y_train)

#Making Prediction
y_pred = knn.predict(X_test)

#Evaluating the Model
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print("Classification Report:\n", classification_rep)

'''- Experiment with different hyperparameters (e.g., changing the value of k in KNN) and use techniques like cross-validation to find the best model configuration.
- Once you are satisfied with the model's performance, you can deploy it to make predictions on new data.'''

# Saving the Model
import joblib

# Save the model to a file
joblib.dump(knn, 'iris_classifier_model.pkl')

# Using Model For Predictions
loaded_model = joblib.load('iris_classifier_model.pkl')
new_data = np.array([[5.1, 3.5, 1.4, 0.2]])  # Replace with your new data
new_data = scaler.transform(new_data)  # Apply the same scaling as before
prediction = loaded_model.predict(new_data)
print(f"Predicted class: {iris.target_names[prediction][0]}")





