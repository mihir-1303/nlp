
import joblib
import pandas as pd

# Load the saved model
model = joblib.load("late_delivery_model.joblib")

# Initial warning threshold
THRESHOLD = 0.4

# Example order details
new_order = pd.DataFrame([{
    "Days for shipment (scheduled)": 2,
    "Order Item Quantity": 2,
    "Order Item Product Price": 100.0,
    "Order Item Discount": 10.0,
    "Order Item Discount Rate": 0.10,
    "Latitude": 12.9716,
    "Longitude": 77.5946,
    "Type": "DEBIT",
    "Shipping Mode": "Standard Class",
    "Market": "Pacific Asia",
    "Order Region": "Southeast Asia",
    "Category Name": "Cleats",
    "Department Name": "Footwear",
    "Customer Segment": "Consumer"
}])

# Predict probability
probability = model.predict_proba(new_order)[0][1]

# Apply our selected warning threshold
prediction = int(probability >= THRESHOLD)

print(f"Estimated late-delivery risk probability: {probability:.2%}")

if prediction == 1:
    print("Warning: Potential late-delivery risk detected.")
else:
    print("No late-delivery warning from the model.")

print(f"Threshold used: {THRESHOLD}")
