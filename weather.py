import requests
import os
from dotenv import load_dotenv

load_dotenv()

def get_weather(city: str) -> dict:
    """Fetch weather data from OpenWeatherMap API"""
    api_key = os.getenv("OPENWEATHER_API_KEY")
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
    
    response = requests.get(url)
    
    if response.status_code == 200:
        data = response.json()
        return {
            "city": city,
            "temperature": data["main"]["temp"],
            "description": data["weather"][0]["description"],
            "humidity": data["main"]["humidity"]
        }
    else:
        return {"error": f"Failed to fetch weather for {city}"}