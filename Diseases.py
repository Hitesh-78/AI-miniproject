import tkinter as tk
from tkinter import messagebox
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Step 1: Load the dataset
dataset = pd.read_csv("disease_dataset.csv")  # Replace with your dataset file

# Step 2: User Interface
def predict_disease():
    age = int(age_entry.get())
    gender = gender_var.get()
    symptoms = symptoms_entry.get()

    # Step 3: Feature Extraction
    X = dataset.drop("Disease", axis=1)
    y = dataset["Disease"]

    # Step 4: Model Training
    model = RandomForestClassifier()
    model.fit(X, y)

    # Step 5: Predictions
    new_data = pd.DataFrame({
        "Age": [age],
        "Gender": [gender],
        "Symptoms": [symptoms]
    })

    prediction = model.predict(new_data)
    probability = model.predict_proba(new_data)[:, 1]

    # Step 6: Display Results
    messagebox.showinfo("Disease Prediction Result", f"Prediction: {prediction[0]}\nProbability: {probability[0]}")

# UI setup
window = tk.Tk()
window.title("Disease Prediction")

age_label = tk.Label(window, text="Age:")
age_label.grid(row=0, column=0, padx=10, pady=10)
age_entry = tk.Entry(window)
age_entry.grid(row=0, column=1, padx=10, pady=10)

gender_label = tk.Label(window, text="Gender:")
gender_label.grid(row=1, column=0, padx=10, pady=10)
gender_var = tk.StringVar(window)
gender_var.set("Male")
gender_dropdown = tk.OptionMenu(window, gender_var, "Male", "Female")
gender_dropdown.grid(row=1, column=1, padx=10, pady=10)

symptoms_label = tk.Label(window, text="Symptoms:")
symptoms_label.grid(row=2, column=0, padx=10, pady=10)
symptoms_entry = tk.Entry(window)
symptoms_entry.grid(row=2, column=1, padx=10, pady=10)

predict_button = tk.Button(window, text="Predict", command=predict_disease)
predict_button.grid(row=3, column=0, columnspan=2, padx=10, pady=10)

window.mainloop()
