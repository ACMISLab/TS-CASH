from collections import deque


class StackHelper:
    """
    A stack helper for first input and first output FIFO.



    """

    def __init__(self):
        self.stack = deque()

    def append(self, item):
        self.stack.append(item)
        return self

    def pop(self):
        """
        None means the stack is empty.

        Returns
        -------

        """
        if self.size() > 0:
            return self.stack.pop()
        else:
            return None

    def size(self):
        return len(self.stack)


if __name__ == '__main__':
    stack = StackHelper()
    stack.append(1).append(2).append(3).append(4).append(5)
    assert stack.size() == 5
    assert stack.pop() == 5
    assert stack.size() == 4
    assert stack.pop() == 4
    assert stack.size() == 3
    assert stack.pop() == 3
    assert stack.pop() == 2
    assert stack.pop() == 1
    assert stack.size() == 0
