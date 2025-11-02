import os


class UtilEnv:
    @staticmethod
    def get_env(key) -> str:
        env_val = os.getenv(key, default=None)
        if env_val is None:
            raise RuntimeError(f"Environment {key} is not set")
        else:
            return env_val
