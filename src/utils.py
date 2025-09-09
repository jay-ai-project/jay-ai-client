import base64
import httpx


def read_base64_from_file_path(path: str) -> str:
    with open(path, "rb") as file:
        return base64.b64encode(file.read()).decode("utf-8")


def read_base64_from_url(url: str) -> str:
    response = httpx.get(url)
    return base64.b64encode(response.read()).decode("utf-8")
