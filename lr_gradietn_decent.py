import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df =  pd.read_csv('Walmart_Sales.csv')
df = df.dropna()

# aggregate sales by date
agg = df.groupby('Date', as_index=False).agg({'Weekly_Sales':'sum'})
# Parse dates robustly: allow day-first formats and coerce invalid strings to NaT
agg['Date'] = pd.to_datetime(agg['Date'], dayfirst=True, errors='coerce')
# drop rows with invalid dates or missing sales
bad_dates = agg['Date'].isna().sum()
if bad_dates:
    print(f"Warning: {bad_dates} invalid dates could not be parsed — dropping those rows.")
    # show a few problematic strings for diagnosis
    bad_rows = agg.loc[agg['Date'].isna()].head(10)
    print(bad_rows)
    agg = agg.dropna(subset=['Date', 'Weekly_Sales'])

# sort by date (important when using ordinal as X)
agg = agg.sort_values('Date')

# Convert dates to days since first date and scale to [0,1] to avoid very large X values
min_date = agg['Date'].min()
days = (agg['Date'] - min_date).dt.days.astype(float)
max_days = days.max() if days.max() > 0 else 1.0
# X as numpy array scaled to [0,1]
X = (days / max_days).astype(float).values

# y as numpy array and scale to [0,1] to avoid large targets
raw_y = agg['Weekly_Sales'].astype(float).values
if np.all(raw_y == 0):
    raise ValueError('All target values are zero.')
y = raw_y / raw_y.max()
print(f"Data loaded: {len(X)} samples. y scaled by max={raw_y.max():.3f}.")

m = 0.0 #slope
b = 0.0 #intercept
# reduce learning rate for stability
lr = 2e-4
epoch = 10000
n = len(X)
losses = []

for _ in range(epoch):
    y_pred = m * X + b #y^​=mx+b

    dm = (-2 / n) * np.sum(X * (y - y_pred)) #gradient of the loss w.r.t slope
    db = (-2 / n) * np.sum(y - y_pred) #gradient of the loss w.r.t intercept

    #MSE
    loss = np.mean((y - y_pred) ** 2)
    # guard against numerical issues
    if not np.isfinite(loss):
        print('Numerical instability detected (loss not finite). Try reducing `lr` or rescaling data.')
        break
    losses.append(loss)

    m -= lr * dm
    b -= lr * db

print(f"Slope: {m}, Intercept: {b}")


# Predict for one day after the last date (scaled)
next_day = agg['Date'].max() + pd.Timedelta(days=1)
X_new = ((next_day - min_date).days) / max_days
pred_scaled = m * X_new + b
# convert prediction back to original y units
pred = pred_scaled * raw_y.max()
print(f"Prediction for next day (scaled X={X_new:.4f}): {pred} (original units)")

plt.figure()
plt.scatter(X, y, label='Data')
x_line = np.linspace(X.min(), X.max(), 100)
y_line = m * x_line + b
plt.plot(x_line, y_line, color='red', label='Fitted line')
plt.legend()
plt.title('Data and fitted line')

plt.figure()
plt.plot(range(1, epoch + 1), losses)
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('Training Loss')
plt.grid(True)
plt.show()

