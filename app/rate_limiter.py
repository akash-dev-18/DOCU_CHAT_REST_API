from slowapi import Limiter
from slowapi.util import get_remote_address
from fastapi import Request


# rate limiting by api-key
def get_api_key(request: Request) -> str:
    return request.headers.get("x-api-key", request.client.host)


limiter = Limiter(key_func=get_api_key)
