# coding=utf-8


import logging.config


class MyLog(object):
    """Provide the project logger."""
    logging.config.fileConfig('./log/logger.conf')
    logger = logging.getLogger('fileConsole')

    @classmethod
    def print(cls, content):
        cls.logger.info(content)
