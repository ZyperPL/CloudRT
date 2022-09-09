from flask import Flask, request
import api.location
import api.weather
import json 
import datetime
from cache import Cache

ERROR_EXTERNAL_CONNECTION_FAILED = -1
ERROR_LOCATION_NOT_FOUND = -2

app = Flask(__name__)

@app.route('/weather', methods=['GET'])
def get_weather():
    lat = request.args.get('lat', 0.0)
    lon = request.args.get('lon', 0.0)

    location_res = None
    if 'location' in request.args:
        print(f"Looking for {request.args['location']}...")
        location_res = api.location.get_location(request.args.get('location'))
    elif 'lat' not in request.args or 'lon' not in request.args:
        print(f"Looking for auto-location...")
        location_res = api.location.get_location("") # auto
    
    if location_res and location_res.status_code == 200:
        location_json = location_res.json()['results'][0]
        lon = location_json['lon']
        lat = location_json['lat']
    
    res_json = None
    cache_entry = Cache.get(lat, lon)
    if cache_entry:
        res_json = cache_entry.data

    if res_json is None:
        res = api.weather.get_weather(lat, lon)
        code = 0
        if res:
            code = res.status_code
            if res.status_code == 200:
                res_json = res.json()
                if res_json:
                   Cache.add(lat, lon, res_json)

    if res_json is not None:
        return res_json
    else:
        print(f"Forcing cache for {lat}, {lon}...")
        cache_entry = Cache.force_get(lat, lon)
        if cache_entry:
            res_json = cache_entry.data

        if res_json is not None:
            return res_json

    return json.dumps({ 'error': code })


LOCATION_CACHE = {}

@app.route('/location', methods=['GET'])
def get_location():
    print(request.args)
    query = request.args.get('query', "")
    print(f"Query: {query}")
    cached_locations = LOCATION_CACHE.get(query, None)
    if not cached_locations:
        res = api.location.get_location(query)
        code = 0
        if res:
            code = res.status_code

        if res is not None and res.status_code == 200:
            res_json = res.json()
            count = res_json['count']
            if count <= 0:
                return json.dumps({ 'error': ERROR_LOCATION_NOT_FOUND })
            LOCATION_CACHE[query] = res_json
            return res_json

        if code == 0:
            code = ERROR_EXTERNAL_CONNECTION_FAILED
        return json.dumps({ 'error': code })

    print(f"Query {query} found in cache.")
    return cached_locations
