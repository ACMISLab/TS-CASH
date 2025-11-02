import multiprocessing
import os


class Stack:
    def __init__(self):
        self.stack = []

    def push(self, item):
        self.stack.append(item)

    def pop(self):
        if not self.is_empty():
            return self.stack.pop()

    def is_empty(self):
        return len(self.stack) == 0

    def remove_by_id(self, item_id):
        for i in range(len(self.stack) - 1, -1, -1):
            if self.stack[i].id == item_id:
                return self.stack.pop(i)
        return None

    def size(self):
        return len(self.stack)


def worker(process_id):
    print("Worker:", process_id, os.getpid())


if __name__ == '__main__':
    import multiprocessing


    def process_task(task, gpu_id):
        # 执行任务的逻辑
        print("Processing task:", task, "on GPU:", gpu_id)


    if __name__ == '__main__':
        tasks = range(1000)
        num_gpus = 8
        max_tasks_per_gpu = 3

        processes = []

        for i, task in enumerate(tasks):
            gpu_id = i % num_gpus  # 根据任务索引计算对应的显卡ID
            p = multiprocessing.Process(target=process_task, args=(task, gpu_id))
            p.start()
            processes.append(p)

            # 如果达到每块显卡最大并行任务数，等待所有任务完成后再继续添加新任务
            if (i + 1) % (num_gpus * max_tasks_per_gpu) == 0:
                for p in processes:
                    p.join()
                processes = []

        for p in processes:
            p.join()
