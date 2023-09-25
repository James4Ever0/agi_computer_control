beat_server_address = dict(host="0.0.0.0", port=8981, beat_url="/beat_request")
beat_client_data = dict(url = (
    f"http://localhost:{beat_server_address['port']}{beat_server_address['beat_url']}"
),timeout=2, access_time_key = 'access_time')

import requests

def heartbeat_base(params):
    r = requests.get(beat_client_data['url'], params = params, timeout=beat_client_data['timeout'])
    assert r.status_code == 200
    data = r.json()
    access_time = data[beat_client_data['access_time_key']]
    return access_time
