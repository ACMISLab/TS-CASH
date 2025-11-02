import requests
from pylibs.utils.util_log import get_logger
import json

log = get_logger()


def http_json_get(url):
    """
    Returns the JSON representation.

    Parameters
    ----------
    url :

    Returns
    -------

    """
    try:
        req = requests.get(url)
        if req.status_code == 200:
            metric_data = json.loads(req.content)
            return metric_data
        else:
            log.error(f"Server response is {req.status_code}, expected 200.")
            return None
    except Exception as e:
        log.error(f"Url {url} is not available. \n{e.args}")
        return None


class Response:
    @staticmethod
    def generate_response(*, code=0, data=[], msg=None):
        return {
            "code": code,
            "data": data,
            "msg": msg
        }

    @staticmethod
    def ok(msg="success", data=None):
        return Response.generate_response(code=0, msg=msg, data=data)

    @staticmethod
    def error(msg, data=None):
        return Response.generate_response(code=-1, msg=msg, data=data)
