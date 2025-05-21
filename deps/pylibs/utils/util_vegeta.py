from dataclasses import dataclass
import os
from pylibs.utils.util_bash import Bash


@dataclass
class Vegeta:

    def attack(self):
        cmd = """
cat > payload.json <<EOF
{
  "n": 32
}
EOF

vegeta attack -rate=1 -duration=1s -timeout 0 <<EOF | vegeta report --type=json
GET your_server_ip:31802
EOF
    """
        Bash.run_command_print_progress(cmd)

    @staticmethod
    def ns_to_s(df):
        """将vageta的原始单位nanoseconds转为s，1ns = 1/1e9s"""

        keys = "duration,latencies_total,latencies_mean,latencies_50th,latencies_90th,latencies_95th,latencies_99th,latencies_max,latencies_min,wait"

        for _k in keys.split(","):
            if _k.strip() != "":
                df[_k] = df[_k] / 1e9
        return df


if __name__ == '__main__':
    # Vegeta().attack()
    Vegeta().attack()
