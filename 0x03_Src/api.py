import requests

def get_weather_forecast(latitude, longitude):
    
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
        data = response.json()
        return data
    else:
        print(f"Error: {response.status_code}")
        return None

def display_forecast(forecast_data):
    if forecast_data:
        daily = forecast_data['daily']
        time = daily['time']
        max_temps = daily['temperature_2m_max']
        min_temps = daily['temperature_2m_min']
        weather_codes = daily['weathercode']
        precipitation_sum = daily['precipitation_sum']
        wind_speed_max = daily['windspeed_10m_max']
        wind_direction = daily['winddirection_10m_dominant']
        sunrise = daily['sunrise']
        sunset = daily['sunset']
        uv_index_max = daily['uv_index_max']
        precipitation_hours = daily['precipitation_hours']
        
        print("Weather forecast for the next 7 days:\n")
        for i in range(len(time)):
            date = time[i]
            max_temp = max_temps[i]
            min_temp = min_temps[i]
            weather_code = weather_codes[i]
            precip_sum = precipitation_sum[i]
            wind_speed = wind_speed_max[i]
            wind_dir = wind_direction[i]
            sun_rise = sunrise[i]
            sun_set = sunset[i]
            uv_index = uv_index_max[i]
            precip_hours = precipitation_hours[i]
            
            # Convert weather code to a description
            weather_desc = get_weather_description(weather_code)
            
            print(f"Date: {date}")
            print(f"Max Temperature: {max_temp}°C")
            print(f"Min Temperature: {min_temp}°C")
            print(f"Weather: {weather_desc}")
            print(f"Precipitation Sum: {precip_sum} mm")
            print(f"Max Wind Speed: {wind_speed} km/h")
            print(f"Dominant Wind Direction: {wind_dir}°")
            print(f"Sunrise: {sun_rise}")
            print(f"Sunset: {sun_set}")
            print(f"UV Index Max: {uv_index}")
            print(f"Precipitation Hours: {precip_hours} hours")
            print("-" * 40)
    else:
        print("No forecast data available.")

def get_weather_description(weather_code):
    # Map weather codes to descriptions (based on Open-Meteo's WMO codes)
    weather_mapping = {
        0: "Clear sky",
        1: "Mainly clear",
        2: "Partly cloudy",
        3: "Overcast",
        45: "Fog",
        48: "Depositing rime fog",
        51: "Light drizzle",
        53: "Moderate drizzle",
        55: "Dense drizzle",
        56: "Light freezing drizzle",
        57: "Dense freezing drizzle",
        61: "Slight rain",
        63: "Moderate rain",
        65: "Heavy rain",
        66: "Light freezing rain",
        67: "Heavy freezing rain",
        71: "Slight snow fall",
        73: "Moderate snow fall",
        75: "Heavy snow fall",
        77: "Snow grains",
        80: "Slight rain showers",
        81: "Moderate rain showers",
        82: "Violent rain showers",
        85: "Slight snow showers",
        86: "Heavy snow showers",
        95: "Thunderstorm",
        96: "Thunderstorm with slight hail",
        99: "Thunderstorm with heavy hail"
    }
    return weather_mapping.get(weather_code, "Unknown weather")

# Coordinates for Oujda, Morocco
latitude = 34.6810  # Latitude for Oujda
longitude = -1.9078  # Longitude for Oujda

# Get the weather forecast
forecast_data = get_weather_forecast(latitude, longitude)

# Display the forecast
display_forecast(forecast_data)