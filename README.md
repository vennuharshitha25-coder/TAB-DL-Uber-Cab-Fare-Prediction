# 🚖 Uber/Ola Fare Prediction Using Machine Learning

## 📌 Project Overview

This project predicts the estimated fare price for cab rides such as Uber and Ola using Machine Learning.
The model analyzes trip-related factors like pickup/drop location, distance, time, traffic conditions, surge pricing, demand level, weather, and ride category to estimate the fare amount.

This helps users get an approximate fare before booking a ride.

---

## 🎯 Objective

To build an intelligent fare prediction system that can:

* Predict ride fare accurately
* Help users compare expected ride costs
* Improve travel budget planning
* Demonstrate real-world Machine Learning application

---

## 🧠 Machine Learning Approach

This project uses **Supervised Learning (Regression)** because the output is a continuous numeric value (fare amount).

### Steps Followed:

1. **Data Collection**

   * Used dataset containing trip and fare details.

2. **Data Preprocessing**

   * Removed unnecessary columns
   * Converted categorical text data into numbers using Label Encoding
   * Handled structured inputs for training

3. **Feature Selection**
   Input features used:

   * Pickup Latitude
   * Pickup Longitude
   * Dropoff Latitude
   * Dropoff Longitude
   * Passenger Count
   * Distance (km)
   * Hour
   * Weekday
   * Weekend Flag
   * Surge Multiplier
   * Traffic Level
   * Demand Level
   * Ride Category
   * Weather

4. **Model Training**

   * Used Random Forest Regressor

5. **Model Saving**

   * Saved trained model as `model.pkl`

6. **Deployment**

   * Built interactive web app using Streamlit

---

## 🤖 Why Random Forest?

Random Forest was selected because:

* High prediction accuracy
* Handles nonlinear relationships well
* Works great with mixed data types
* Reduces overfitting
* Reliable for fare prediction datasets

---

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Joblib
* Streamlit

---

## 📂 Project Structure

```text
UberFairPredictions/
│──dataset/ uberdatasett.csv
│── train_model.py
│── app.py
│── model.pkl
│── encoders.pkl
│── features.pkl
│── README.md
```

---

## ▶️ How to Run the Project

### Step 1: Install Required Libraries

```bash
pip install pandas numpy scikit-learn streamlit joblib
```

---

### Step 2: Train the Model

```bash
python train_model.py
```

This will generate:

* `model.pkl`
* `encoders.pkl`
* `features.pkl`

---

### Step 3: Run the Web App

```bash
python -m streamlit run app.py
```

---

### Step 4: Open in Browser

Streamlit automatically opens:

```text
http://localhost:8501
```

---

## 💰 Sample Output

Input:

* Distance = 12 km
* Traffic = High
* Surge = 1.5
* Ride Type = Prime

Output:

```text
Estimated Fare: ₹325.75
```

---

## 📈 Future Enhancements

* Compare Uber vs Ola prices separately
* Real-time traffic integration
* Google Maps API route estimation
* Mobile app version
* Dynamic surge forecasting

---

## ✅ Conclusion

This project demonstrates how Machine Learning can solve real-world transportation pricing problems. By using trip data and ride conditions, the system predicts fare prices efficiently and can be expanded into a smart travel assistant.

---

## 👩‍💻 Author

Harshitha
Harini
Suhas
