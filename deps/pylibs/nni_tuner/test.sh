cd "$( cd "$( dirname "$0" )" >/dev/null 2>&1 && pwd )"
nnictl create -c  debug_lhs_rns.yaml -f  --port 9999 -d
