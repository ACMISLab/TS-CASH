import hashlib
import zlib


def generate_32bit_hashcode(input_string):
    # 将输入字符串转换为bytes，因为zlib.crc32需要bytes类型的输入
    input_bytes = input_string.encode('utf-8')

    # 计算输入字符串的CRC32哈希值
    hashcode = zlib.crc32(input_bytes)

    # CRC32产生的是一个无符号的32位整数，直接返回这个值
    return hashcode


class UtilHash:

    @staticmethod
    def get_str_hash(string):
        return get_str_hash(string)

    @staticmethod
    def get_crc32_hash(string):
        """
        生成32位的字符串hash值
        Parameters
        ----------
        string :

        Returns
        -------

        """
        return str(generate_32bit_hashcode(string))

def get_str_hash(string):
    """
    Get the hash code of string

    Parameters
    ----------
    string : str
        The input str

    Returns
    -------
    str
        The hash code of the string

    """
    if not isinstance(string,str):
        raise ValueError(f"get_str_hash excepts str, but recept {type(string)}")
    return hashlib.sha1(string.encode("utf8")).hexdigest()
