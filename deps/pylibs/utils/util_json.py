import json

from pylibs.utils.util_message import logw


def is_json(data):
    try:
        json.loads(data)
    except Exception as e:
        logw(e)
        return False

    return True
