import io
import os.path

import requests

from pylibs.utils.util_directory import make_dirs


class WebUtil:
    @staticmethod
    def download_pdf(pdf_url, pdf_abs_name):
        """
        Down a pdf to a file with  pdf_abs_name
        Parameters
        ----------
        pdf_url :
        pdf_abs_name :

        Returns
        -------

        """
        make_dirs(os.path.dirname(pdf_abs_name))
        bytes_io = io.BytesIO(requests.get(pdf_url).content)
        with open(pdf_abs_name, "wb") as f:
            f.write(bytes_io.getvalue())
        return pdf_abs_name


if __name__ == '__main__':
    WebUtil.download_pdf("https://www.vldb.org/pvldb/vol16/p1790-martens.pdf", "runtime/a.pdf")
