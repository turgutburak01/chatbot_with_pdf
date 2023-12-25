from fastapi import FastAPI

from api.main import api_router

from fastapi.middleware.cors import CORSMiddleware

import logging
import uvicorn
from helper.config import Config
from util.logger import MyLogger
from logging import DEBUG


config = Config()
app = FastAPI()

logging.basicConfig(level=logging.INFO)


app.logger = MyLogger("api", level=DEBUG)
app.logger.info("Main API is running now")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



app.secret_key = config.APP_SECRET_KEY
app.include_router(api_router)
   
    
# Press the green button in the gutter to run the script.
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=config.PORT)
