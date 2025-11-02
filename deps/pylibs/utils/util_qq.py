import time

import requests
import json


def send_msg_to_qq(msg):
    url = "http://your_server_ip:12000/qq/send"
    payload = json.dumps({
        "group_id": "778177226",
        "msg": msg
    })
    headers = {
        'Content-Type': 'application/json'
    }
    response = requests.request("POST", url, headers=headers, data=payload)
    return response.status_code
