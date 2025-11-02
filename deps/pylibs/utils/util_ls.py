import re


class UtilLS:
    """
    A util for linux command: ls
    """
    DIR_PATTERN = r'd.*?\s+\d+\s+[a-zA-Z]+\s[a-zA-Z]+ \d+ [a-zA-Z]+\s\d+:\d+ ([^\x1b]+)'

    @staticmethod
    def get_all_dir_name(ls_result: str):
        matches = re.findall(UtilLS.DIR_PATTERN, ls_result)

        # Print the matched directory names
        for match in matches:
            print(match)


if __name__ == '__main__':
    text = "drwxrwxrwx 11 root root 4096 Aug 25 13:15 \x1b[34;42mv920_02_fastuts_vus_roc_0.001_random_daphnet_ecg_iops_mgab_opportunity_smd_svdb_yahoo\x1b[0m\x1b[K'"

    # Define the regular expression pattern to match the directory name

    # Find all matches of the pattern in the text
    UtilLS.get_all_dir_name(text)
