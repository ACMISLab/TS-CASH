import traceback
import click
from pylibs.utils.util_log import get_logger

log = get_logger()


@click.command()
def main():
    print("Ok")


if __name__ == '__main__':
    try:
        main(standalone_mode=False)
    except Exception as e:
        traceback.print_exc()
        log.error(traceback.format_exc())
