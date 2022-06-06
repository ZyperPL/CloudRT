from flask import Flask, request
import api.location
import api.weather
import json 

ERROR_EXTERNAL_CONNECTION_FAILED = -1
ERROR_LOCATION_NOT_FOUND = -2

app = Flask(__name__)

@app.route('/weather', methods=['GET'])
def get_weather():
    lat = request.args.get('lat', 0.0)
    lon = request.args.get('lon', 0.0)

    location_res = None
    if 'location' in request.args:
        location_res = api.location.get_location(request.args.get('location'))
    else:
        location_res = api.location.get_location("") # auto
    
    if location_res and location_res.status_code == 200:
        location_json = location_res.json()['results'][0]
        lon = location_json['lon']
        lat = location_json['lat']
    

    res = api.weather.get_weather(lat, lon)

    code = 0
    if res:
        code = res.status_code

    if res is not None and res.status_code == 200:
        return res.json()

    return json.dumps({ 'error': code })


@app.route('/location', methods=['GET'])
def get_location():
    query = request.args.get('query', "")
    res = api.location.get_location(query)
    code = 0
    if res:
        code = res.status_code

    if res is not None and res.status_code == 200:
        res_json = res.json()
        if res_json['count'] <= 0:
            return json.dumps({ 'error': ERROR_LOCATION_NOT_FOUND })
        return res_json['results'][0]

    if code == 0:
        code = ERROR_EXTERNAL_CONNECTION_FAILED
    return json.dumps({ 'error': code })
