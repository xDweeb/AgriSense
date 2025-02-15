import json
import numpy as np
import requests
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
import joblib
import threading
import time

# For RL, we use stable-baselines3 and a custom gym environment
import gym
from gym import spaces
from stable_baselines3 import SAC

# ===============================
# 1. Weather API functions
# ===============================
def get_weather_forecast(latitude: float, longitude: float):
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        'latitude': latitude,
        'longitude': longitude,
        'daily': [
            'temperature_2m_max', 'temperature_2m_min', 'weathercode',
            'precipitation_sum', 'windspeed_10m_max', 'winddirection_10m_dominant',
            'sunrise', 'sunset', 'uv_index_max', 'precipitation_hours'
        ],
        'timezone': 'auto',
        'forecast_days': 7  
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        return response.json()
    else:
        return None

# ===============================
# 2. Define a custom RL environment
# ===============================
class RealIrrigationEnv(gym.Env):
    """
    A custom Gym environment for real-time RL training.
    The observation is a vector containing:
      [current_soil_moisture, current_temperature, current_rainfall, current_NDVI,
       SVR_predicted_Q, SVR_predicted_T]
    The action space allows small adjustments to the SVR predictions.
    """
    def _init_(self):
        super(RealIrrigationEnv, self)._init_()
        # Define observation: 6 values
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0, 0, 0]),
            high=np.array([100, 50, 50, 1.0, 100, 24]),
            dtype=np.float32
        )
        # Action: adjust Q by [-1,1] mm and T by [-0.5, 0.5] hours
        self.action_space = spaces.Box(
            low=np.array([-1, -0.5]),
            high=np.array([1, 0.5]),
            dtype=np.float32
        )
        self.state = None

    def step(self, action):
        # In a real environment, the new state would be determined by physics.
        # Here, we simulate an update for online training purposes.
        # Our reward is based on how close the soil moisture is to the target range (30-60).
        current_sm, temp, rainfall, NDVI, svr_Q, svr_T = self.state

        # Apply RL adjustments (action)
        adjusted_Q = svr_Q + action[0]
        adjusted_T = svr_T + action[1]

        # Simulate new soil moisture:
        # For instance, assume soil moisture increases with irrigation and rainfall,
        # and decreases with high temperature.
        new_sm = current_sm + adjusted_Q + rainfall - (0.3 * temp)
        new_sm = np.clip(new_sm, 0, 100)

        # Simulate NDVI update (simple model)
        new_NDVI = NDVI + (0.01 * adjusted_Q) - (0.005 * temp)
        new_NDVI = np.clip(new_NDVI, 0, 1.0)

        # Form new state: assume temperature and rainfall remain similar
        self.state = np.array([new_sm, temp, rainfall, new_NDVI, adjusted_Q, adjusted_T], dtype=np.float32)

        # Reward: optimal if soil moisture in [30,60]
        if 30 <= new_sm <= 60:
            reward = 1.0
        else:
            reward = -abs(new_sm - 45) / 10

        done = False  # Continuous training
        return self.state, reward, done, {}

    def reset(self, state=None):
        # Reset with the provided state (from sensor/forecast data)
        if state is not None:
            self.state = state
        else:
            # Default random initialization
            self.state = np.array([50, 25, 10, 0.5, 7, 30], dtype=np.float32)
        return self.state

# ===============================
# 3. Load Pretrained Models
# ===============================
# Load pre-trained SVR models and scalers
svr_volume = joblib.load("svr_volume.pkl")
svr_duration = joblib.load("svr_duration.pkl")
scaler_X = joblib.load("scaler_X.pkl")
scaler_y_volume = joblib.load("scaler_y_volume.pkl")
scaler_y_duration = joblib.load("scaler_y_duration.pkl")

# Load or initialize RL model
# For demonstration, we assume an RL model already exists.
# Otherwise, you can initialize a new one:
rl_env = RealIrrigationEnv()
rl_model = SAC("MlpPolicy", rl_env, verbose=0)
# Optionally load a saved model:
# rl_model = SAC.load("sac_irrigation", env=rl_env)

# ===============================
# 4. Define FastAPI app and Request Model
# ===============================
app = FastAPI()

class SensorData(BaseModel):
    soil_moisture: float
    temperature: float
    rainfall: float
    NDVI: float
    latitude: float  = 34.6810  # Default for Oujda
    longitude: float = -1.9078  # Default for Oujda

@app.post("/predict")
def predict_irrigation(sensor: SensorData):
    """
    Expects sensor data every 7 hours.
    Gets weather forecast, computes initial SVR predictions,
    then refines with the RL agent.
    Also uses this observation for online RL training.
    """
    # 1. Get latest weather forecast
    weather_data = get_weather_forecast(sensor.latitude, sensor.longitude)
    if not weather_data:
        raise HTTPException(status_code=500, detail="Weather data fetch failed")

    # For simplicity, use first day's forecast values:
    daily = weather_data['daily']
    forecast_temp_max = daily['temperature_2m_max'][0]
    forecast_rainfall = daily['precipitation_sum'][0]

    # 2. Build feature vector for SVR:
    # For demonstration, we combine sensor readings with forecast values.
    # In your preprocessing pipeline, you should arrange features exactly as in training.
    # Here, we assume the SVR expects a flat vector.
    X_input = np.array([
        sensor.soil_moisture,
        sensor.temperature,
        sensor.rainfall,
        sensor.NDVI,
        forecast_temp_max,
        forecast_rainfall
    ]).reshape(1, -1)

    # Standardize input features
    X_input_scaled = scaler_X.transform(X_input)

    # 3. Get initial predictions from SVR
    Q_pred_scaled = svr_volume.predict(X_input_scaled)[0]
    T_pred_scaled = svr_duration.predict(X_input_scaled)[0]

    # Inverse-transform if needed (here we assume scaler_y_* were used during training)
    Q_pred = scaler_y_volume.inverse_transform([[Q_pred_scaled]])[0][0]
    T_pred = scaler_y_duration.inverse_transform([[T_pred_scaled]])[0][0]

    # 4. Build the state for RL
    # Our RL environment state: [soil_moisture, temperature, rainfall, NDVI, SVR_Q, SVR_T]
    state = np.array([
        sensor.soil_moisture,
        sensor.temperature,
        sensor.rainfall,
        sensor.NDVI,
        Q_pred,
        T_pred
    ], dtype=np.float32)

    # Reset RL environment with the new state
    rl_env.reset(state=state)

    # 5. Use RL model to predict adjustment action
    action, _ = rl_model.predict(state)
    # Apply action to adjust predictions:
    Q_final = Q_pred + action[0]
    T_final = T_pred + action[1]
    # Enforce non-negative irrigation (and reasonable bounds)
    Q_final = max(Q_final, 0.1)
    T_final = max(T_final, 0.5)

    # 6. (Optional) Use this new experience for online RL training.
    # Here we simulate one step of training using our current environment.
    # In practice, you'd accumulate experiences in a replay buffer.
    rl_model.learn(total_timesteps=500, reset_num_timesteps=False)

    # 7. Return the final irrigation prediction
    result = {
        "SVR_prediction": {"irrigation_volume": Q_pred, "irrigation_duration": T_pred},
        "RL_adjustment": {"delta_volume": action[0], "delta_duration": action[1]},
        "Final_decision": {"irrigation_volume": Q_final, "irrigation_duration": T_final},
        "weather": {
            "forecast_temperature_max": forecast_temp_max,
            "forecast_rainfall": forecast_rainfall
        }
    }
    return result

# ===============================
# 5. Background Scheduler for Periodic Tasks (Optional)
# ===============================
# If you need to trigger predictions automatically every 7 hours,
# you could run a background task that calls the prediction endpoint.
# Here’s an example using threading:
def periodic_prediction():
    while True:
        # Simulate sensor data. In production, fetch from your sensors.
        sensor_data = {
            "soil_moisture": np.random.uniform(20, 60),
            "temperature": np.random.uniform(15, 35),
            "rainfall": np.random.uniform(0, 10),
            "NDVI": np.random.uniform(0.3, 0.9),
            "latitude": 34.6810,
            "longitude": -1.9078
        }
        try:
            response = requests.post("http://localhost:8000/predict", json=sensor_data)
            print("Periodic Prediction:", response.json())
        except Exception as e:
            print("Error during periodic prediction:", e)
        # the requsts shculded every 7 hours
        time.sleep(7 * 3595)

# Uncomment the following lines to start the periodic prediction in a separate thread.
# threading.Thread(target=periodic_prediction, daemon=True).start()

# ===============================
# 6. Run the FastAPI App
# ===============================
if _name_ == "_main_":
    uvicorn.run(app, host="0.0.0.0", port=8000)