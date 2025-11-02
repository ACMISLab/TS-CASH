import re


def parse_cmd_to_dict(cmd):
    """


    Parameters
    ----------
    cmd :  str
        python models/deepsvdd.py  --m_dataset SHUTTLE  --seed 0 --data_sample_method RS --data_sample_rate  1.0


    Returns
    -------
    dict
        {'python': 'models/deepsvdd.py', 'm_dataset': 'SHUTTLE', 'seed': '0', 'data_sample_method': 'RS', 'data_sample_rate': '1.0'}

    """

    res = cmd.split("--")
    ret = {}
    for line in res:
        line_ = re.sub("\s+", " ", line)
        line_ = line_.strip()
        if str(line_).startswith('-'):
            raise ValueError("Unsupported type for single -(e.g., -user 1)")
        line_arr = line_.split(" ")
        if len(line_arr) == 1:
            line_arr.append(" ")
        assert len(line_arr) >= 2
        assert line_[0] != " "
        ret[line_arr[0]] = line_arr[1]
    return ret


def parse_str_to_float(str_float) -> float:
    """
    Convert a string to a float, supporting the following format: 0.25, 1, or 1/54

    Parameters
    ----------
    str_float :

    Returns
    -------

    """
    if isinstance(str_float, float):
        return str_float

    if isinstance(str_float, int):
        return float(str_float)

    try:
        if str(str_float).find("/") > -1:
            d_arr = str_float.split("/")
            if len(d_arr) != 2:
                raise ValueError("input is not supported, support 0.25, 1, or 1/54")
            return float(d_arr[0]) / float(d_arr[1])
        else:
            return float(str_float)
    except:
        raise ValueError("input is not supported, support 0.25, 1, or 1/54")
