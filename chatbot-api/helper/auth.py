from fastapi import Depends, HTTPException
from fastapi.security import HTTPBasicCredentials, HTTPBasic
from starlette import status

from helper.config import Config
from dotenv import load_dotenv

load_dotenv()
config = Config()

security = HTTPBasic()

def verification(creds: HTTPBasicCredentials = Depends(security)):
    username = creds.username
    password = creds.password


    stored_username = config.API_BASIC_AUTH_USERNAME
    stored_password = config.API_BASIC_AUTH_PASSWORD

    is_username = username == stored_username
    is_password = password == stored_password

    print(is_username)
    print(is_password)

    if is_username and is_password:
        return True

    else:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Basic"},
        )