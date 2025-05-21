import abc


class Dataset(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def get_source_train_data(self):
        """
        Get source_the training data without labels.

        Returns
        -------

        """
        pass

    @abc.abstractmethod
    def get_source_train_missing(self):
        """
        Get source_the missing indicator of the training data.

        Returns
        -------

        """
        pass

    @abc.abstractmethod
    def get_source_train_label(self):
        """
        Get source_the training labels  without data.

        Returns
        -------

        """
        pass

    @abc.abstractmethod
    def get_source_valid_data(self):
        """
        Get source_the valid data without labels

        Returns
        -------

        """
        pass

    @abc.abstractmethod
    def get_source_valid_missing(self):
        """
        Get source_the missing indicator of the valid data

        Returns
        -------

        """
        pass

    @abc.abstractmethod
    def get_source_test_missing(self):
        """
        Get source_the missing indicator of the test data

        Returns
        -------

        """
        pass

    @abc.abstractmethod
    def get_source_valid_label(self):
        """
        Get source_the valid labels  without data

        Returns
        -------

        """
        pass

    @abc.abstractmethod
    def get_source_test_label(self):
        """
        Get source_the test labels  without data

        Returns
        -------

        """
        pass

    @abc.abstractmethod
    def get_source_test_data(self):
        """
        Get source_the test data  without labels

        Returns
        -------

        """
        pass

    @abc.abstractmethod
    def get_sliding_windows_three_splits(self):
        """
        Get source_the training data with sampling_method and sampling_rate,
        Return x,train_label,test_x,test_label

        Sampling:
        1. Grab all sliding windows of training data
        2. Sampling training windows from the training windows
        3. Remove the anomaly windows
        4. Return all testing windows of the test data.


        Returns
        -------

        """
        pass

    @abc.abstractmethod
    def get_source_train_timestamp(self):
        """
        Get source_the timestamp of the training data

        Parameters
        ----------
        Returns
        -------

        """
        pass

    @abc.abstractmethod
    def get_source_test_timestamp(self):
        """
        Get source_the timestamp of the test data

        Parameters
        ----------
        Returns
        -------

        """
        pass

    @abc.abstractmethod
    def get_source_valid_timestamp(self):
        """
        Get source_the timestamp of the valid data

        Parameters
        ----------
        Returns
        -------

        """
        pass
