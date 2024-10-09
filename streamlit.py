import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

# Load dataset (Replace this with your actual dataset path)
url = "https://raw.githubusercontent.com/RoshanManick/Heart-attack-prediction-using-ML/main/Heart%20Attack%20Data%20Set.csv"
data = pd.read_csv(url)

# Prepare the data
X = data.drop(columns='target')
y = data['target']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Evaluate the model
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"Model Accuracy: {accuracy:.2f}")

# Save the model
joblib.dump(model, 'heart_attack_model.pkl')
