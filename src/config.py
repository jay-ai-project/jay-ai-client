from starlette.config import Config

env = Config()


OLLAMA_BASE_URL = env("OLLAMA_BASE_URL", cast=str, default="http://localhost:2024")
