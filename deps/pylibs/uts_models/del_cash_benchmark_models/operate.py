import argparse
from pylibs.utils.util_log import get_logger

log = get_logger()
parser = argparse.ArgumentParser()
type_helps = """
example: list all examples script:
    python pca/examples/pca.py
    python ocsvm/examples/examples.py
    python hbos/examples/hbos.py
    python ae/examples/examples.py
    python random_forest/examples/examples.py
    python iforest/examples/iforest.py
    python cnn/examples/examples.py
    python lof/examples/lof.py
    python vae/examples/examples.py
    python lstm/examples/examples.py
    
clear_imgs: list all generated images


"""
from pylibs.utils.util_bash import exec_cmd

parser.add_argument("--type", help=type_helps, default="example")
parser.add_argument("--arg1", type=str, required=False)
parser.add_argument("--arg2", type=str, required=False)
args = parser.parse_args()

from pylibs.utils.util_file import FileUtils

import pdfkit


def convert_md_to_pdf(md_file, pdf_file):
    try:
        pdfkit.from_file(md_file, pdf_file)
        print("转换成功！")
    except Exception as e:
        print("转换失败：", str(e))


if __name__ == '__main__':
    if args.type == 'example':
        all = FileUtils().get_all_files("./", ext=".py")
        for f in all:
            if str(f).find("example") > -1:
                print("python " + f.split("benchmark_models/")[-1])
    elif args.type == 'clear_imgs':
        all = FileUtils().get_all_files("./", ext=".png")
        for f in all:
            if str(f).find("plot_images") > -1:
                exec_cmd("rm -f " + f.split("benchmark_models/")[-1])
    elif args.type == "pdf":
        convert_md_to_pdf(args.arg1, args.arg2)
    else:
        log.warning(f"Unknown type of argument:{args.type}")
