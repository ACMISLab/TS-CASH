from pyutils.kvdb.kvdb_sqlite import KVDBSqlite

dbsqlite = KVDBSqlite(dbfile="/Users/sunwu/SW-Research/AutoML-Benchmark/tshpo/tshpo_alg_perf.sqlite")
dbsqlite.query({"kjsd": "jsldk"})
