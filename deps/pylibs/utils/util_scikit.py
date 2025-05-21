import os.path

from joblib import dump, load
from numpy.testing import assert_almost_equal

from pylibs.utils.util_common import UtilComm
from pylibs.utils.util_directory import make_dirs

from pylibs.utils.util_log import get_logger

log = get_logger()


class UtilSK:
    @staticmethod
    def save_model(model, model_path, home=UtilComm.get_runtime_directory()):
        """
         from sklearn import svm
        from sklearn import datasets
        clf = svm.SVC()
        X, y= datasets.load_iris(return_X_y=True)
        clf.fit(X, y)

        UtilSK.save_model(clf,"aaa.joblib")
        clf=UtilSK.load_model("aaa.joblib")

        Parameters
        ----------
        model : class
            The sklearn model
        model_path :
        home :

        Returns
        -------

        """

        if not model_path.endswith(".joblib"):
            model_path = model_path + ".joblib"
        model_path = os.path.join(home, model_path)
        
        make_dirs(home)
        UtilSys.is_debug_mode() and log.info(f"Model is saved to {os.path.abspath(model_path)}")
        dump(model, model_path)
        return model_path

    @staticmethod
    def load_model(model_path):
        """

        Parameters
        ----------
        model_path :

        Returns
        -------

        """
        if not model_path.endswith(".joblib"):
            model_path = model_path + ".joblib"
        return load(model_path)


if __name__ == '__main__':
    from sklearn import svm
    from sklearn import datasets

    clf = svm.SVC()
    X, y = datasets.load_iris(return_X_y=True)
    clf.fit(X, y)
    print(clf.class_weight_)

    model_path = UtilSK.save_model(clf, "svc")
    clf1 = UtilSK.load_model(model_path)
    print(clf1.class_weight_)

    assert_almost_equal(clf.class_weight_, clf1.class_weight_)
