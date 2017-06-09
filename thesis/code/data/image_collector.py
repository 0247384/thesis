import requests
from io import BytesIO
from PIL import Image

mapbox_url = 'https://api.mapbox.com/styles/v1/mapbox/satellite-v9/static/'
bing_url = 'https://dev.virtualearth.net/REST/v1/Imagery/Map/Aerial/'
google_url = 'https://maps.googleapis.com/maps/api/staticmap'

mapbox_key = '' # TODO
bing_key = '' # TODO
google_key = '' # TODO


def get_image(center, zoom, size, source):
    lat = center.lat
    lon = center.lon

    if source == 'mapbox':
        url = mapbox_url + '{},{},{}/{}x{}'.format(lon, lat, zoom, size, size)
        payload = {'access_token': mapbox_key, 'logo': 'false', 'attribution': 'false'}
    elif source == 'bing':
        zoom += 1
        url = bing_url + '{},{}/{}'.format(lat, lon, zoom)
        payload = {'mapSize': '{},{}'.format(size, size), 'format': 'png', 'key': bing_key}
    elif source == 'google':
        size = int(size / 2)
        url = google_url
        payload = {'center': '{},{}'.format(lat, lon), 'zoom': zoom, 'size': '{}x{}'.format(size, size), 'scale': 2,
                   'format': 'png32', 'maptype': 'satellite', 'key': google_key}
    else:
        raise ValueError('Image source not supported!')

    response = requests.get(url, params=payload)

    print('Status code:', response.status_code)
    if response.status_code != 200:
        raise Exception(response.text)
    else:
        return Image.open(BytesIO(response.content))
