import logging
from datetime import datetime
from logging import INFO
import uuid
from util.constant import LogConstants
from starlette.requests import Request
import json


class CustomJsonFormatter(logging.Formatter):
    def format(self, record):
        log_data = {
            "times": self.formatTime(record),
            "level": record.levelname,
            "name": record.name,
            "message": record.getMessage()
        }
        return json.dumps(log_data)


class MyLogger(object):
    def __init__(self, name, level=INFO):
        self.level = level
        self.name = str(name)
        self.console_logger = logging.StreamHandler()
        self.console_logger.set_name(name)
        self.formatter = CustomJsonFormatter()
        self.console_logger.setFormatter(self.formatter)
        self.logger = logging.getLogger(self.name)
        self.logger.setLevel(self.level)
        if not self.logger.hasHandlers():
            self.logger.addHandler(self.console_logger)

    def info(self, msg, extra={}):
        self.logger.info(msg, extra=extra)

    def error(self, msg, extra={}, exc_info=True):
        self.logger.error(msg, extra=extra, exc_info=exc_info)

    def start_operation(self, operation_name):
        return OperationLog(operation_name, self).add_field(LogConstants.CONTEXT_ID, str(uuid.uuid4()))

    def start_service_operation(self, operation_name, request: Request):
        return OperationLog(operation_name, self, request).add_field(LogConstants.CONTEXT_ID, str(uuid.uuid4()))


class OperationLog(object):
    def __init__(self, operation_name, logger: MyLogger, request: Request = None):
        self.__start_time = datetime.now()
        self.__logger = logger
        self.__log = {LogConstants.OPERATION_NAME: operation_name}
        self.__add_ip_address(request)

    def add_field(self, field, value):
        self.__log[field] = value
        return self

    def __get_start_time(self):
        return self.__start_time

    def __add_ip_address(self, request: Request):
        if request:
            ip_address = request.headers.get('x-real-ip', None)
            if not ip_address:
                ip_address = request.client.host
            self.add_field(LogConstants.IP_ADDRESS, ip_address)

    def succeed(self):
        self.add_field(LogConstants.OPERATION_STATUS, LogConstants.OPERATION_SUCCEEDED)
        self.__add_operation_duration()
        self.__logger.info(self.__log)

    def fail(self, exc_info=True):
        self.add_field(LogConstants.OPERATION_STATUS, LogConstants.OPERATION_FAILED)
        self.__add_operation_duration()
        self.__logger.error(self.__log, exc_info=exc_info)

    def __add_operation_duration(self):
        self.__log[LogConstants.OPERATION_TOOK] = round((datetime.now() - self.__start_time).total_seconds() * 1000, 2)
