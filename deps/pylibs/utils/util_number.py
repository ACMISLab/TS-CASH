import numpy as np


def float_str_to_fraction_str(float_number):
    """
    Converts a float number to a fraction str.

    E.g., convert '0.0078125' to '$\\frac{1}{64}$'


    # return the math representation for the fraction, e.g.,  '$\\frac{1}{64}$'

    Parameters
    ----------
    float_number :

    Returns
    -------

    """
    float_number = float(float_number)
    if float_number == 0:
        return "0"
    elif float_number == 1:
        return "1"
    else:

        return f"1/{int(1 / float(float_number))}"


def float_str_to_math(float_number):
    """
    Converts a float number to a fraction str.

    E.g., convert '0.0078125' to '$\\frac{1}{64}$'


    # return the math representation for the fraction, e.g.,  '$\\frac{1}{64}$'

    Parameters
    ----------
    float_number :

    Returns
    -------

    """
    float_number = float(float_number)
    if float_number == 0:
        return "$0$"
    elif float_number == 1:
        return "1"
    else:

        return f"$\\frac{{1}}{{{int(1 / float(float_number))}}}$"


def str_to_float(str_number: str):
    """
    Convert 1/2 to 0.5
    """
    if str(str_number).find("/") > -1:
        numbers = str_number.split("/")
        float_number = float(numbers[0]) / float(numbers[1])
        return float_number
    else:
        return float(str_number)


def is_number(value):
    # 尝试将变量转换为浮点数，如果不是数字则会引发异常
    try:
        float(value)
        return True
    except:
        return False


# def round_dict(data: dict, round=4):
#     for key in data.keys():
#         _val = data[key]
#         if is_number(_val):
#             data[key] = np.round(_val, 4)
#     return data

def round_dict(data, decimals=4, excludes=None):
    if excludes is None:
        excludes = []

    for key in data.keys():
        if key in excludes:
            continue
        if is_number(data[key]):
            data[key] = np.round(data[key], decimals)

    return data


if __name__ == '__main__':
    print(str_to_float("1/2"))

    print(round_dict({
        "a": 3.22222222222,
        "b": "sdfs"
    }))
