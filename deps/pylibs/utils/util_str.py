import hashlib
import json
import re


def clear_str(txt):
    """
    Clear txt
    Parameters
    ----------
    str :

    Returns
    -------

    """
    return re.sub('\\x1b', '', txt).replace("[0m", "").replace("\n")


def get_str_letters(input_str):
    """
    只保留字符串中的英文字母
    :param input_str:
    :type input_str:
    :return:
    :rtype:
    """

    # s = "这是1个示例字符串! 包含English letters, 数字123和sfs符号#*。"
    s = re.sub(r'[^a-zA-Z]', '', input_str)  # 移除非字母字符
    return s


def get_set_from_string(input_str):
    return set([str(i).lower() for i in input_str.split(" ")])


def get_list_from_string(input_str) -> list:
    return [str(i).lower() for i in input_str.split(" ")]


def get_common_letters(ccf_title: str, paper_title: str) -> list:
    # 将列表转换为集合
    ccf_title = get_list_from_string(ccf_title)
    paper_title = get_list_from_string(paper_title)

    result = []

    # 遍历第一个列表，检查元素是否在第二个列表中
    for a in ccf_title:
        if a in paper_title:
            result.append(a)
    # 如果需要，可以将结果转换回列表
    return result


def json_str_to_dict(json_str):
    return json.loads(json_str)


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
    if not isinstance(string, str):
        raise ValueError(f"get_str_hash excepts str, but recept {type(string)}")
    return hashlib.sha1(string.encode("utf8")).hexdigest()


def get_str_hash_sha256(string):
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
    if not isinstance(string, str):
        raise ValueError(f"get_str_hash excepts str, but recept {type(string)}")
    return hashlib.sha3_256(string.encode("utf8")).hexdigest()


def str_to_hash(string):
    return get_str_hash(string)


def str2hash(string):
    return get_str_hash(string)
