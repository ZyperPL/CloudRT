from datetime import datetime
from datetime import timedelta
import json

class CacheEntry:
    TIMESTAMP_DELTA = 86400

    def __init__(self, lat, lon, data):
        self.lat = lat
        self.lon = lon
        self.data = data
        self.timestamp = datetime.timestamp(datetime.today())

    def valid(self, other_ts):
        delta = other_ts - self.timestamp
        print(f"Delta time: {delta} ({other_ts} - {self.timestamp})")
        return abs(other_ts - self.timestamp) < CacheEntry.TIMESTAMP_DELTA

class Cache:
    CACHE = None

    def load():
        def unmap_entry(ej):
            entry = CacheEntry(ej['lat'], ej['lon'], ej['data'])
            entry.timestamp = ej['timestamp']
            return entry

        print("Loading data...\n")
        Cache.CACHE = {}
        try:
            with open("cache.json", "r") as f:
                json_data = json.load(f)
                for e in json_data:
                    entry = unmap_entry(e['entry'])
                    Cache.CACHE[e['key'][0], e['key'][1]] = entry
        except Exception as e:
            print(e)


    def save():
        def remap_entry(entry: CacheEntry):
            return { "lat": entry.lat, "lon": entry.lon, "data": entry.data, "timestamp": entry.timestamp }

        def remap_cache(cache):
            remapped = []
            for k, v in Cache.CACHE.items():
                remapped.append({"key": k, "entry": remap_entry(v) })
            return remapped

        cache_remapped = remap_cache(Cache.CACHE)
        with open("cache.json", "w") as f:
            json.dump(cache_remapped, f)


    def get(lat, lon):
        if Cache.CACHE is None:
            Cache.load()

        current_timestamp = datetime.timestamp(datetime.today())
        entry = Cache.CACHE.get((lat, lon), None)

        if not entry or not entry.valid(current_timestamp):
            return None
 
        print("Entry found in cache.")
        return entry
    
    def force_get(lat, lon):
        if Cache.CACHE is None:
            Cache.load()
        
        print("Entry found in cache.")
        return Cache.CACHE.get((lat, lon), None)

    def add(lat, lon, data):
        entry = CacheEntry(lat, lon, data)
        Cache.CACHE[lat, lon] = entry
        print("Entry added to cache.")
        Cache.save()
