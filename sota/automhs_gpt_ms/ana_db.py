from sota.auto_cash.auto_cash_helper import KVDB

kvdb = KVDB("authchs_gpt.dump", "./")
print(kvdb.values())
