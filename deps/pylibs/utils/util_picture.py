import os
import warnings

from PIL import Image
from fpdf import FPDF
from pylibs.utils.util_log import get_logger

log = get_logger()


class UtilPic:
    @staticmethod
    def merge_picture(home, file_name):
        """


        Parameters
        ----------
        home : str
            aaa/path/
        out_path :
            bbb/a.pdf

        Returns
        -------

        """
        if not file_name.endswith(".pdf"):
            file_name = file_name + ".pdf"
        path = home

        imagelist = []
        for i in os.listdir(path):
            if i.endswith(".png") or i.endswith(".jpg"):
                imagelist.append(i)
        img_path = os.path.join(path, imagelist[0])
        image = Image.open(img_path)
        pdf = FPDF(unit="pt", format=[image.width, image.height])
        pdf.set_font('Arial', 'B', 16)
        pdf.set_auto_page_break(0)  # 自动分页设为False
        for image in sorted(imagelist):
            pdf.add_page()
            img_path = os.path.join(path, image)
            if img_path.endswith(".png") or img_path.endswith(".jpg"):

                # pdf.image(img_path, w=image.width, h=image.height)  # 指定宽高

                pdf.image(img_path, 0, 100)  # 指定宽高
                pdf.cell(0, 0, os.path.basename(img_path))
            else:
                warnings.warn(f"Unsupported file type for image: {img_path}")
                continue
        path = os.path.join(path, file_name)
        UtilSys.is_debug_mode() and log.info(f"Merage pdf is saved to {os.path.abspath(path)}")
        pdf.output(path, "F")


if __name__ == '__main__':
    UtilPic.merge_picture(
        "/Users/sunwu/SW-Research/sw-research-code/A01_paper_exp/experiment_repro/runtime/plot_images/ECG",
        "summary.pdf")
