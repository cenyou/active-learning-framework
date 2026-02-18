# Copyright (c) 2024 Robert Bosch GmbH
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import logging
from datetime import datetime

class FormattedWrapper(object):
    def __init__(self, logger):
        self.logger = logger

    def __getattr__(self, attr):
        if attr in ['debug', 'info', 'warning', 'error', 'critical']:
            return super().__getattr__(attr)
        else:
            return getattr(self.logger, attr)

    def debug(self, msg, *args, **kwargs):
        self.logger.debug('\n' + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + ' - ' + str(msg), *args, **kwargs)

    def info(self, msg, *args, **kwargs):
        self.logger.info(
            '\n\x1b[32;20m' + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + ' - ' + str(msg) + '\x1b[0m', *args, **kwargs
        )
    def warning(self, msg, *args, **kwargs):
        self.logger.warning(
            '\n\x1b[33;20m' + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + ' - ' + str(msg) + '\x1b[0m', *args, **kwargs
        )
    def error(self, msg, *args, **kwargs):
        self.logger.error(
            '\n\x1b[31;20m' + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + ' - ' + str(msg) + '\x1b[0m', *args, **kwargs
        )
    def critical(self, msg, *args, **kwargs):
        self.logger.critical(
            '\n\x1b[31;1m' + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + ' - ' + str(msg) + '\x1b[0m', *args, **kwargs
        )

def getLogger(*args, **kwargs):
    return FormattedWrapper(logging.getLogger(*args, **kwargs))


if __name__ == '__main__':
    logger = getLogger(__name__)
    logging.basicConfig(level=logging.DEBUG)
    logger.debug('test debug')
    logger.info('test info')
    logger.warning('test warning')
    logger.error('test error')
    logger.critical('test critical')
