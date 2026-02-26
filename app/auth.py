import os
from fastapi import Security, HTTPException, status
from fastapi.security import APIKeyHeader

api_key_header = APIKeyHeader(name="PDF-CHAT-API-KEY", auto_error=False)


async def verify_api_key(api_key: str = Security(api_key_header)) -> str:
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key missing.",
        )

    if api_key != os.getenv("API_KEY"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API key.",
        )

    return api_key
