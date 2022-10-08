import requests

url_1h = 'http://my.meteoblue.com/packages/clouds-1h_wind-1h_air-1h'
url_day = 'http://my.meteoblue.com/packages/clouds-day_wind-day_air-day'

def get_weather(lat: float, lon: float):
    params = {
        'lat': lat,
        'lon': lon,
        'tz': 'Europe/Warsaw',
        'format': 'json',
        'temperature': 'C',
        'history_days': 3,
        'forecast_days': 7,
        'timeformat': 'timestamp_utc',
        'apikey': '3ovr7uFjKx4gUGpR'
    }
    try:
        params_safe = params.copy()
        params_safe['apikey'] = 'X'*15
        print(f"Making request to {url_1h} params={params_safe}...")
        res = requests.get(url_1h, params=params)
        return res
    except Exception as exception:
        print(exception)

    return None

