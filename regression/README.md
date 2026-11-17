# Uber Fare Prediction with a Decision Tree Regressor

Predict the **fare (price) of an Uber ride** in New York from where it starts, where it ends, when it happens and how many passengers are in the car.

- **Type of problem:** regression (the answer is a number in dollars)
- **Model:** `DecisionTreeRegressor` (scikit-learn), `max_depth=8`
- **Data:** `uber.csv`, 200,000 rides
- **Result:** test R2 **0.84**, average error (MAE) about **$1.99**

---

## 1. Files in this folder

| File | What it is |
|---|---|
| `regression.ipynb` | The notebook with every step: cleaning, training, testing, plots |
| `uber.csv` | The raw data (200,000 rides) |
| `uber_model.pkl` | The trained model, saved so you can use it without training again |
| `README.md` | This file |

---

## 2. The data

`uber.csv` has 9 columns:

| Column | Meaning | Used for training? |
|---|---|---|
| `Unnamed: 0` | row ID | no (dropped) |
| `key` | ride ID with a time stamp | no (dropped) |
| `fare_amount` | **the price in dollars (the target we predict)** | target |
| `pickup_datetime` | date and time of pickup | yes (turned into hour, day, weekday, month, year) |
| `pickup_longitude`, `pickup_latitude` | where the ride starts | yes |
| `dropoff_longitude`, `dropoff_latitude` | where the ride ends | yes |
| `passenger_count` | number of passengers | yes |

---

## 3. What the notebook does, step by step

### Step 1. Load and look
Read `uber.csv`, check the shape (200,000 rows, 9 columns), the empty values and the min and max of each column.
The data has errors: a latitude of 1,644 (impossible), a fare of 499, and 208 passengers.

### Step 2. Drop columns that are not useful
`Unnamed: 0` and `key` are only labels. If they stay, the tree can memorize them instead of learning real patterns.

### Step 3. Clean the bad rows
| Rule | Why |
|---|---|
| remove rows with empty values | 1 row has an empty dropoff location |
| keep fares above 0 and up to 100 | removes zero or negative fares and extreme fares |
| keep 1 to 6 passengers | removes rides with 0 passengers and the 208 passenger error |
| keep coordinates in the New York area (latitude 40 to 42, longitude -75 to -72) | removes impossible locations |

About 195,000 rows remain.

### Step 4. Make columns from the date
A tree cannot use a date written as text, so the notebook creates: `hour`, `day`, `weekday` (Monday = 0), `month`, `year`.

### Step 5. Make the distance column (the most important one)
The file has only longitude and latitude, but the fare depends mostly on **how far** the ride goes.
The haversine formula turns the two points into a straight-line distance in kilometers (`distance_km`).

### Step 6. Remove impossible distances
Rides with a distance of 0 (cancelled or wrong rides) and rides above 100 km are removed.
About **192,988 rides** remain.

### Step 7. Split into train and test
- `X` = the 11 input columns, `y` = `fare_amount`
- 80% of the rides are for learning (train), 20% are kept hidden for testing
- `random_state=42` makes the split the same every time

### Step 8. Baseline (the score to beat)
If you always guess the average fare, you are off by about **$5.95** on average. The tree must do much better than this.

### Step 9. Tree with no limit (to see overfitting)
A tree allowed to grow without limit reaches depth 45 and scores **R2 = 1.0 on train** but only about **0.74 on test**.
It memorized the training rides instead of learning the pattern. This is called **overfitting**.

### Step 10. Find the best `max_depth`
The notebook tries depths 1 to 15 with 5-fold cross-validation and picks the best one: **depth 8**.
The score rises up to depth 8 and then falls, because deeper trees start to overfit.

### Step 11. Final model
`DecisionTreeRegressor(max_depth=8, random_state=42)` trained on the training data.

### Step 12. Evaluate
| Metric | Value | Meaning |
|---|---|---|
| Test R2 | **0.8432** | the model explains about 84% of the fare differences |
| Train R2 | 0.8662 | close to test R2, so only a little overfitting |
| MAE | **1.986** | the prediction is off by about $2 on average |
| RMSE | 3.742 | higher than MAE, because a few rides have big errors |
| Within 10% | 35.0% | share of rides predicted within 10% of the real fare |
| Within 20% | 63.9% | share of rides predicted within 20% of the real fare |
| Baseline MAE | 5.947 | the tree's error is about one third of this |

### Step 13. Plots
- **Actual vs predicted:** points near the red line are good predictions.
- **The tree drawing:** the top 3 levels, with the question in each box and the average fare of the rides in it.
- **Feature importance:** which columns the tree used most.

### Feature importance (what matters for the fare)
| Column | Importance |
|---|---|
| `distance_km` | **0.926** |
| `dropoff_longitude` | 0.027 |
| `year` | 0.023 |
| everything else | less than 0.01 each |

The distance does almost all the work. `year` matters a little because fares changed over the years.

---

## 4. How to run it

### First time: create the environment (Linux or Mac)
```bash
cd path/to/this/folder
python3 -m venv .venv
source .venv/bin/activate
pip install jupyter pandas numpy matplotlib scikit-learn joblib
jupyter notebook
```
On Windows, activate with `.venv\Scripts\activate`.

### Every other time
```bash
cd path/to/this/folder
source .venv/bin/activate
jupyter notebook
```

Then open `regression.ipynb`, click **Kernel -> Restart Kernel and Run All Cells**, and make sure `uber.csv` is in the same folder.

---

## 5. Using the saved model (`uber_model.pkl`)

The file contains the trained tree and the list of 11 columns in the right order.
A new ride must be prepared in the same way as in training (date parts and distance), so the code below does that first.

```python
import joblib
import numpy as np
import pandas as pd

saved = joblib.load('uber_model.pkl')
model = saved['model']

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = p2 - p1
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlambda / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))

def prepare_ride(raw_df, saved):
    d = raw_df.copy()
    d['pickup_datetime'] = pd.to_datetime(d['pickup_datetime'])
    d['hour'] = d['pickup_datetime'].dt.hour
    d['day'] = d['pickup_datetime'].dt.day
    d['weekday'] = d['pickup_datetime'].dt.day_of_week
    d['month'] = d['pickup_datetime'].dt.month
    d['year'] = d['pickup_datetime'].dt.year
    d['distance_km'] = haversine_km(d['pickup_latitude'], d['pickup_longitude'],
                                    d['dropoff_latitude'], d['dropoff_longitude'])
    return d[saved['columns']]

new_ride = {
    'pickup_datetime': '2015-06-15 18:30:00 UTC',
    'pickup_longitude': -73.9855, 'pickup_latitude': 40.7580,      # Times Square
    'dropoff_longitude': -73.7781, 'dropoff_latitude': 40.6413,    # JFK airport
    'passenger_count': 1,
}

ride = prepare_ride(pd.DataFrame([new_ride]), saved)
print("Predicted fare: $", round(model.predict(ride)[0], 2))
```

Example results from the tested model:

| Ride | Distance | Predicted fare |
|---|---|---|
| short | 0.9 km | about $5.85 |
| medium | 9.1 km | about $27.84 |
| long (Times Square to JFK) | 21.8 km | about $55.53 |

The fare rises with the distance, as it should.

---

## 6. Checks to run before saving the model

Run the "Pre-save check" cell. It prints `OK` or `PENDING` for each of these:

- no empty values and no text columns in `X`
- `Unnamed: 0`, `key` and `fare_amount` are not in `X`
- fares between 0 and 100, passengers between 1 and 6
- coordinates inside the New York area, distance between 0 and 100 km
- `X` has exactly 11 columns
- tree depth is 8 or less
- train and test R2 are close (gap below 0.05) and test R2 is above 0.80
- the model beats the baseline

Save only when it prints `ALL DONE, safe to save`.

After saving, check that:
- the loaded model gives the same predictions as the original on `X_test`
- raw rides run through `prepare_ride` give the same predictions as the training columns
- a longer ride gets a higher fare

---

## 7. Mistakes that were found and fixed (good to remember)

| Mistake | Effect | Fix |
|---|---|---|
| `df[df['fare_amount'] > 0 & (df['fare_amount'] < 100)]` | In Python `&` is evaluated before `>`, so the "less than 100" part was ignored and fares up to 499 stayed. Test R2 dropped to 0.77. | give each condition its own brackets: `(a > 0) & (b <= 100)` |
| `print(df.drop(['Unnamed: 0'], axis=1))` | It only prints. The column stayed in the data. | `df = df.drop(columns=['Unnamed: 0', 'key'])` |
| `round(x <= 10).mean() * 100` | The bracket was in the wrong place, so the result was not rounded. | `round((x <= 10).mean() * 100, 1)` |

---

## 8. Words used in this project

| Word | Simple meaning |
|---|---|
| **Regression** | predicting a number (here, a price) |
| **Decision tree** | a set of yes/no questions that ends in an answer, for example "is the distance below 3 km?" |
| **Feature** | an input column, such as `distance_km` |
| **Target** | the column we predict (`fare_amount`) |
| **Train / test** | train = rides the model learns from, test = hidden rides used to check it |
| **Overfitting** | the model memorizes the training data and does badly on new data |
| **max_depth** | the maximum number of questions in a row. Smaller means a simpler tree. |
| **Cross-validation** | testing several times on different parts of the training data to get a fair score |
| **MAE** | average size of the error in dollars |
| **RMSE** | like MAE, but big errors count more |
| **R2** | how much of the fare variation the model explains (1.0 is perfect, 0 is no better than guessing the average) |
| **Baseline** | the score you get by always guessing the average. The model must beat it. |
| **pkl file** | a saved Python object, here the trained model |

---

## 9. Limits of this model

- It only knows rides like the ones it saw: New York area, 1 to 6 passengers, fares up to $100. Outside this range its answers are not reliable.
- The distance is a straight line, but real trips follow roads, and traffic and tolls are not in the data.
- A tree predicts in steps, so a very small change in the distance can give the same fare.
- `year` is an input. Use realistic dates (the data covers 2009 to 2015).
- Load the pkl file with the **same scikit-learn version** used to save it, and only load pkl files you trust.

---

## 10. Ideas for the next step

- Compare with a Random Forest Regressor, which usually scores a bit higher on this data.
- Add more columns, for example whether the ride goes to an airport.
- Build a small form (for example with Streamlit) where you type the pickup and dropoff and get the fare.
