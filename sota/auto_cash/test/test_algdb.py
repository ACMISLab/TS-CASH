from sota.auto_cash.auto_cash_helper import KVDB

# {'alpha': 0.07020513, 'average': 'False', 'fit_intercept': 'True', 'learning_rate': 'invscaling', 'loss': 'perceptron', 'penalty': 'l1', 'tol': 2.023e-05, 'eta0': 5.1e-07, 'power_t': 0.76285027, '__choice__': 'sgd', 'random_state': 42, 'model': 'sgd', 'dataset': 'electricity', 'fold_index': 4}
kvdb = KVDB()
for i in kvdb.keys():
    print(f"\n{i}")
# print(kvdb.query(kvdb.keys()[0]))
