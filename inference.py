import joblib
import sys

# Load the model
model = joblib.load("model.joblib")

# Take input from command line
input_text = " ".join(sys.argv[1:])
prediction = model.predict([input_text])

print("Recommendation:", prediction[0])