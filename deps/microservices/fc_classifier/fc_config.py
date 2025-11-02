from pyutils.util_sys import is_macos


def get_classic_data_home():
    if is_macos():
        DATA_HOME = "/Volumes/sw_data/phd_paper_data/ms_dataset/classic"
    else:
        DATA_HOME = "/remote-home4/cs_acmis_xxx/tshpo_ms/ms_dataset/classic"

    return DATA_HOME


def get_ms_data_home():
    if is_macos():
        DATA_HOME = "/Volumes/sw_data/phd_paper_data/ms_dataset"
    else:
        DATA_HOME = "/remote-home4/cs_acmis_xxx/tshpo_ms/ms_dataset"

    return DATA_HOME
