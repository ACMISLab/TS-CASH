import os.path
import sys


class UtilLib:
    @staticmethod
    def add_libs():
        """
        Down a pdf to a file with  pdf_abs_name
        Parameters
        ----------
        pdf_url :
        pdf_abs_name :

        Returns
        -------

        """
        home = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
        sys.path.append(os.path.join(home, "datasets"))
        sys.path.append(os.path.join(home, "py-search-lib"))
        sys.path.append(os.path.join(home, "timeseries-models"))


if __name__ == '__main__':
    UtilLib.add_libs()
