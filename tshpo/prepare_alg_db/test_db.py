from pyutils.kvdb.kvdb_sqlite import KVDBSqlite

dbsqlite = KVDBSqlite(dbfile="/Users/xxx/Research/AutoML-Benchmark/tshpo/tshpo_alg_perf.sqlite")
dbsqlite.query({"kjsd": "jsldk"})
