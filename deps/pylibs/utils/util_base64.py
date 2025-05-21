import base64
import json


class UtilBase64:
    @staticmethod
    def encode_json(json_obj):
        return base64.b64encode(json.dumps(json_obj).encode('utf-8')).decode("utf-8")

    @staticmethod
    def decode_json(b64string)->json:
        _str = base64.b64decode(b64string).decode("utf-8")
        return json.loads(_str)


def base64_encode_str(input_str):
    if input_str is None:
        return None
    return str(base64.b64encode(input_str.encode('utf-8')), "utf-8")


def base64_decode_str(input_str):
    return str(base64.b64decode(input_str), "utf-8")


if __name__ == '__main__':
    string = "hello world"
    assert string == base64_decode_str(base64_encode_str(string))
    print(base64_encode_str("1"))
    print(base64_encode_str("s"))
    print(base64_encode_str("照片没干过"))
    print(base64_encode_str("sdfsdfsd"))
