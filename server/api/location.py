import requests

url = 'https://www.meteoblue.com/en/server/search/query3'

def get_location(query: str):
    params = { 
        'query': query,
        'page': 1,
        'itemsPerPage': 8
    }
    try:
        print(f"Making request to {url} params={params}...")
        res = requests.get(url, params=params)
        return res
    except Exception as exception:
        print(exception)

    return None

