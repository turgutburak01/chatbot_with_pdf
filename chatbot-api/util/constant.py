''' 
    Constants and Enum Classes 
'''
from enum import Enum


class Tags:
    HOST = 'HOST'
    PORT = 'PORT'
    APP_SECRET_KEY = 'APP_SECRET_KEY'
    DB_URI = 'DB_URI'
    NAME = 'name'
    DATA = 'data'

class LogConstants:
    CONTEXT_ID = "context_id"
    OPERATION_NAME = "operation_name"
    OPERATION_STATUS = "operation_status"
    OPERATION_SUCCEEDED = "succeeded"
    OPERATION_FAILED = "failed"
    OPERATION_TOOK = "operation_took"
    NAME = "name"
    DESCRIPTION = "description"
    QUESTION = "question"
    DATA = "data"
    IP_ADDRESS = "ip_address"