Modify pynisher to fix #1591 of auto-sklearn on macOS: https://github.com/automl/auto-sklearn/issues/1591

```
Traceback (most recent call last):
  File "/Users/timothy/.virtualenvs/moml/lib/python3.9/site-packages/autosklearn/automl.py", line 765, in fit
    self._do_dummy_prediction()
  File "/Users/timothy/.virtualenvs/moml/lib/python3.9/site-packages/autosklearn/automl.py", line 489, in _do_dummy_prediction
    raise ValueError(msg)
ValueError: (' Dummy prediction failed with run state StatusType.CRASHED and additional output: {\'error\': \'Result queue is empty\', \'exit_status\': "<class \'pynisher.limit_function_call.AnythingException\'>", \'subprocess_stdout\': \'\', \'subprocess_stderr\': \'Process pynisher function call:\\nTraceback (most recent call last):\\n  File "/usr/local/Cellar/python@3.9/3.9.14/Frameworks/Python.framework/Versions/3.9/lib/python3.9/multiprocessing/process.py", line 315, in _bootstrap\\n    self.run()\\n  File "/usr/local/Cellar/python@3.9/3.9.14/Frameworks/Python.framework/Versions/3.9/lib/python3.9/multiprocessing/process.py", line 108, in run\\n    self._target(*self._args, **self._kwargs)\\n  File "/Users/timothy/.virtualenvs/moml/lib/python3.9/site-packages/pynisher/limit_function_call.py", line 108, in subprocess_func\\n    resource.setrlimit(resource.RLIMIT_AS, (mem_in_b, mem_in_b))\\nValueError: current limit exceeds maximum limit\\n\', \'exitcode\': 1, \'configuration_origin\': \'DUMMY\'}.',)
```
