"""
A helper class to record the progress of a experiment.


For example:
if an experiment is error when running the 30th trial, this class will save an integer to a
file, which indicates the experiment has completed 29 trials with status DONE, and the experiment
will run start_or_restart 30 next time.
"""
import os

from pylibs.utils.util_file import generate_random_file


class ResumeHelper():
    def __init__(self, id):
        self._id_file = generate_random_file(ext=".idx", name=id)

    def record(self, progress: int):
        """
        Save the exp_index.


        Parameters
        ----------
        progress :

        Returns
        -------

        """
        with open(self._id_file, "w") as f:
            f.write(str(progress))

    def get_latest_index(self):
        """
        -1 means not exist

        Returns
        -------

        """
        if not os.path.exists(self._id_file):
            return -1
        with open(self._id_file, "r") as f:
            _index = int(f.read())
        return _index
