import json
import numpy as np
from sklearn.svm import SVR
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
# Load preprocessed JSON data
with open("processed_irrigation_data.json", "r") as f:
    data = json.load(f)

# Sort data by date to preserve time order
data.sort(key=lambda x: x["date"])

# Initialize lists for inputs and outputs
X_past, X_future = [], []
y_volume, y_duration = [], []

for sample in data:
    X_past.append(np.array(sample["past_data"]).flatten())   # Flatten past sequence (365, 8) → (365*8,)
    X_future.append(np.array(sample["future_data"]).flatten())  # Flatten future sequence (7, 6) → (7*6,)
    output = sample["output"]
    y_volume.append(output["irrigation_volume"])
    y_duration.append(output["irrigation_duration"])
    # y_countdown.append(output["irrigation_countdown"])

# Convert lists to NumPy arrays
X_past = np.array(X_past)         # (n_samples, 365*8)
X_future = np.array(X_future)     # (n_samples, 7*6)
X = np.hstack([X_past, X_future]) # Combine past & future features (n_samples, 365*8 + 7*6)

y_volume = np.array(y_volume)
y_duration = np.array(y_duration)
# y_countdown = np.array(y_countdown)

# Chronological Train-Test Split (Reserve most recent 20% as test set)
n_samples = X.shape[0]
train_size = int(n_samples * 0.8)

X_train, X_test = X[:train_size], X[train_size:]
y_volume_train, y_volume_test = y_volume[:train_size], y_volume[train_size:]
y_duration_train, y_duration_test = y_duration[:train_size], y_duration[train_size:]
# y_countdown_train, y_countdown_test = y_countdown[:train_size], y_countdown[train_size:]

# Standardize data (important for SVR)
scaler_X = StandardScaler()
scaler_y_volume = StandardScaler()
scaler_y_duration = StandardScaler()

X_train = scaler_X.fit_transform(X_train)
X_test = scaler_X.transform(X_test)

y_volume_train = scaler_y_volume.fit_transform(y_volume_train.reshape(-1, 1)).ravel()
y_volume_test = scaler_y_volume.transform(y_volume_test.reshape(-1, 1)).ravel()

y_duration_train = scaler_y_duration.fit_transform(y_duration_train.reshape(-1, 1)).ravel()
y_duration_test = scaler_y_duration.transform(y_duration_test.reshape(-1, 1)).ravel()

# Define SVR models
svr_volume = SVR(kernel="rbf", C=5, epsilon=0.05)
svr_duration = SVR(kernel="rbf", C=5, epsilon=0.05)

# K-Fold Cross-Validation (n_splits=5)
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Lists to store training and validation losses
volume_train_losses, volume_val_losses = [], []
duration_train_losses, duration_val_losses = [], []

# Perform cross-validation and track losses
print("\nPerforming cross-validation...")
for train_index, val_index in kf.split(X_train):
    X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
    y_volume_train_fold, y_volume_val_fold = y_volume_train[train_index], y_volume_train[val_index]
    y_duration_train_fold, y_duration_val_fold = y_duration_train[train_index], y_duration_train[val_index]

    # Train SVR models
    svr_volume.fit(X_train_fold, y_volume_train_fold)
    svr_duration.fit(X_train_fold, y_duration_train_fold)

    # Predict on training and validation sets
    volume_train_pred = svr_volume.predict(X_train_fold)
    volume_val_pred = svr_volume.predict(X_val_fold)
    duration_train_pred = svr_duration.predict(X_train_fold)
    duration_val_pred = svr_duration.predict(X_val_fold)

    # Compute MSE for training and validation
    volume_train_loss = mean_squared_error(y_volume_train_fold, volume_train_pred)
    volume_val_loss = mean_squared_error(y_volume_val_fold, volume_val_pred)
    duration_train_loss = mean_squared_error(y_duration_train_fold, duration_train_pred)
    duration_val_loss = mean_squared_error(y_duration_val_fold, duration_val_pred)

    # Append losses to lists
    volume_train_losses.append(volume_train_loss)
    volume_val_losses.append(volume_val_loss)
    duration_train_losses.append(duration_train_loss)
    duration_val_losses.append(duration_val_loss)

# Print average losses
print(f"Average Training Loss - Volume: {np.mean(volume_train_losses):.4f}")
print(f"Average Validation Loss - Volume: {np.mean(volume_val_losses):.4f}")
print(f"Average Training Loss - Duration: {np.mean(duration_train_losses):.4f}")
print(f"Average Validation Loss - Duration: {np.mean(duration_val_losses):.4f}")

# Plot loss curves
plt.figure(figsize=(12, 6))

# Volume Loss Plot
plt.subplot(1, 2, 1)
plt.plot(volume_train_losses, label="Training Loss (Volume)")
plt.plot(volume_val_losses, label="Validation Loss (Volume)")
plt.xlabel("Fold")
plt.ylabel("Mean Squared Error")
plt.title("Volume Loss Curve")
plt.legend()

# Duration Loss Plot
plt.subplot(1, 2, 2)
plt.plot(duration_train_losses, label="Training Loss (Duration)")
plt.plot(duration_val_losses, label="Validation Loss (Duration)")
plt.xlabel("Fold")
plt.ylabel("Mean Squared Error")
plt.title("Duration Loss Curve")
plt.legend()

plt.tight_layout()
plt.show()

# Train on Full Training Data
print("\nTraining final SVR models...")
svr_volume.fit(X_train, y_volume_train)
svr_duration.fit(X_train, y_duration_train)

# Evaluate on Test Set
print("\nEvaluating on test set...")
test_volume_pred = svr_volume.predict(X_test)
test_duration_pred = svr_duration.predict(X_test)

# Compute Mean Squared Error
test_volume_mse = mean_squared_error(y_volume_test, test_volume_pred)
test_duration_mse = mean_squared_error(y_duration_test, test_duration_pred)

print(f"Test Loss - Volume: {test_volume_mse:.4f}")
print(f"Test Loss - Duration: {test_duration_mse:.4f}")