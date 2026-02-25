import os
from fastapi import Security, HTTPException, status
from fastapi.security import APIKeyHeader

api_key_header = APIKeyHeader(name="PDF-CHAT-API-KEY", auto_error=False)

key = os.getenv("API_KEY")


async def verify_api_key(api_key: str = Security(api_key_header)):
    if not key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail=f"MISSING API KEY"
        )

    return api_key
