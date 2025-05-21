import datetime
import json
import os
import pprint
import traceback

import requests

from pylibs.config import config_webhook_of_feishu
from pylibs.utils.util_message import log_warn_msg


class FSUtil:

    @staticmethod
    def send_msg_to_feishu(msg):
        return send_msg_to_feishu(msg)

    @staticmethod
    def report_error_trace_to_feishu(msg=None):
        if msg is None:
            send_msg_to_feishu(f"{datetime.datetime.now()}\nError：\n{pprint.pformat(traceback.format_exc())}")
        else:
            send_msg_to_feishu(
                f"{datetime.datetime.now()}\nError：\n{pprint.pformat(traceback.format_exc())}. \n-----------\n {msg}")


def send_msg_to_feishu(msg):
    # 你复制的webhook地址
    # log.info(f"Send [{str(msg)}] to feishu.")
    msg = f"({os.uname().nodename}) {msg}"
    try:
        url = config_webhook_of_feishu()
        payload_message = {
            "msg_type": "text",
            "content": {
                "text": msg
            }
        }
        headers = {
            'Content-Type': 'application/json'
        }
        response = requests.request("POST",
                                    url,
                                    headers=headers,
                                    data=json.dumps(payload_message),
                                    timeout=5)

        return response.status_code == 200
    except Exception as e:
        log_warn_msg(f"Sending message to feishu is error. cause: {e}")
        traceback.print_exc()


def report_error_trace():
    send_msg_to_feishu(f"{datetime.datetime.now()}\nError：\n{pprint.pformat(traceback.format_exc())}")


def report_error_trace_to_feishu():
    report_error_trace()


def feishu_report_error_and_exit():
    """
    Report the error trace to feishu, then exit the program.

    Returns
    -------
    None

    """
    report_error_trace()
    raise RuntimeError("Exit program by user. ")


def feishu_report_and_exit(message):
    """
    Report the error trace to feishu, then exit the program.

    Returns
    -------
    None

    """
    send_msg_to_feishu(f"{datetime.datetime.now()}\nError：\n{pprint.pformat(message)}")
    raise RuntimeError("Exit program by user. ")


if __name__ == '__main__':
    send_msg_to_feishu("111")
