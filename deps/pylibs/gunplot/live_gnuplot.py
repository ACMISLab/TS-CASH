"""
实时绘制 live.gnu
"""
from pylibs.utils.util_gnuplot import Gnuplot
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
WATCH_FILE = 'live.gnu'

gp = Gnuplot()
def replot(file):
    gp.sets('reset session')
    gp.sets('reset')
    gp.sets('clear')
    with open(file, "r") as f:
        lines = f.readlines()
        gp.sets("\n".join(lines))
        gp.draw()


class FileChangeHandler(FileSystemEventHandler):
    def __init__(self, callback):
        self.callback = callback

    def on_modified(self, event):
        if event.src_path.endswith(WATCH_FILE):
            self.callback(WATCH_FILE)


def function_b(file_path):
    print(f"文件 {file_path} 发生了改变，调用函数b")
    replot(file_path)


if __name__ == "__main__":
    replot(WATCH_FILE)
    event_handler = FileChangeHandler(function_b)
    observer = Observer()
    observer.schedule(event_handler, path='.', recursive=False)
    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
