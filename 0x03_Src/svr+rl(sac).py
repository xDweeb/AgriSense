import json
import numpy as np
from sklearn.svm import SVR
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import gym
from gym import spaces

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


class IrrigationEnv(gym.Env):
    def __init__(self):
        super(IrrigationEnv, self).__init__()

        # Observation: [Soil Moisture, Temperature, Rainfall, NDVI, Initial Q, Initial T]
        self.observation_space = spaces.Box(low=np.array([10, 10, 0, 0.2, 0, 0]),
                                            high=np.array([100, 40, 20, 1.0, 50, 24]),
                                            dtype=np.float32)

        # Action: Adjust Q and T (increase/decrease irrigation volume & duration)
        self.action_space = spaces.Box(low=np.array([-1, -0.5]), high=np.array([1, 0.5]), dtype=np.float32)


        self.reset()

    def step(self, action):
        adj_Q, adj_T = action  # Adjustment to irrigation values
        Q_optimal = self.state[4] + adj_Q  # Adjusted irrigation volume
        T_optimal = self.state[5] + adj_T  # Adjusted irrigation duration

        # Weather impact
        temperature = self.state[1]
        rainfall = self.state[2]

        # Update soil moisture based on irrigation
        soil_moisture = self.state[0] + Q_optimal + rainfall - (0.3 * temperature)
        soil_moisture = np.clip(soil_moisture, 10, 100)

        # Plant Growth (NDVI)
        NDVI_prev = self.state[3]
        NDVI = NDVI_prev + (0.02 * Q_optimal) - (0.005 * temperature)
        NDVI = np.clip(NDVI, 0.2, 1.0)

        # Reward Function
        moisture_reward = 1.5 if 30 <= soil_moisture <= 60 else -abs(soil_moisture - 45) / 5
        growth_reward = (NDVI - NDVI_prev) * 20  # Boost plant growth effect
        water_penalty = -0.02 * max(Q_optimal, 0)  # Lower penalty for reasonable irrigation

        reward = moisture_reward + growth_reward + water_penalty

        # Update state
        self.state = np.array([soil_moisture, temperature, rainfall, NDVI, Q_optimal, T_optimal], dtype=np.float32)

        done = False  # Continuous learning

        return self.state, reward, done, {}

    def reset(self):
        self.state = np.array([np.random.uniform(20, 60),  # Soil Moisture
                               np.random.uniform(10, 35),  # Temperature
                               np.random.uniform(0, 20),  # Rainfall
                               np.random.uniform(0.3, 0.9),  # NDVI
                               np.random.uniform(10, 40),  # Initial Q from SVR
                               np.random.uniform(5, 20)],  # Initial T from SVR
                              dtype=np.float32)
        return self.state

from stable_baselines3 import SAC

# Create environment
env = IrrigationEnv()

# Train RL Model (SAC)
model = SAC("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=1000)

rl_model = model

# Get SVR Prediction
def svr_predict(X_input):
    Q_pred = svr_volume.predict([X_input])[0]
    T_pred = svr_duration.predict([X_input])[0]
    return Q_pred, T_pred

# Example input (past & future features)
example_input = X_train[0]

# Get initial irrigation values from SVR
Q_svr, T_svr = svr_predict(example_input)

# Get RL optimization
initial_state = np.array([5, 25, 10, 0.5, Q_svr, T_svr], dtype=np.float32)
action, _ = rl_model.predict(initial_state)
Q_final = Q_svr + action[0]
T_final = T_svr + action[1]

print(f"SVR Prediction: Q = {Q_svr:.2f} mm, T = {T_svr:.2f} hours")
print(f"RL Optimized Decision: Q = {Q_final:.2f} mm, T = {T_final:.2f} hours")
