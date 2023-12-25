from pydantic_settings import BaseSettings


class Config(BaseSettings):

    OPENAI_API_KEY: str
    APP_SECRET_KEY: str
    API_BASIC_AUTH_USERNAME : str
    API_BASIC_AUTH_PASSWORD : str
    PORT: int
    HOST: str
    DEBUG: bool = False


    class Config:
        env_file = ".env"
