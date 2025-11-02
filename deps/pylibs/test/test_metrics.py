from unittest import TestCase

from pylibs.utils.util_metrics import affiliation_metrics


class TestMetrics(TestCase):
    def test_test(self):
        vector_pred = [0, 0, 0, 0, 1, 0, 0, 0, 1, 0]
        vector_gt = [0, 0, 0, 1, 0, 0, 0, 1, 1, 1]
        prec, recall, f1 = affiliation_metrics(vector_gt, vector_pred)
        assert prec == 0.8181818181818181
        assert recall == 0.8442760942760943
        assert f1 == 2 * prec * recall / (prec + recall)
        print(f"prec:{prec},recall:{recall},f1:{f1}")
        # assert res == {'individual_precision_distances': [0.5, 0.0],
        #                'individual_precision_probabilities': [0.6363636363636364, 1.0],
        #                'individual_recall_distances': [0.5, 0.3333333333333333],
        #                'individual_recall_probabilities': [0.8181818181818181, 0.8703703703703703],
        #                'precision': 0.8181818181818181,
        #                'recall': 0.8442760942760943}
