"""
Main script for collecting a night's worth of data from Beddit sensor
"""

import logging
import tempfile
from datetime import datetime
from beddit import BedditWriter


def start_logging(log_level=logging.INFO):
    """Initialize some very basic logging to a file"""
    log_file = f"beddit_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s')
    logging.basicConfig(filename=log_file, level=log_level)
    logging.info("Starting...")


# create datafile
df = tempfile.TemporaryFile()
start_logging()
writer = BedditWriter(df)
result = writer.scan()
if result:
    writer.receive_until_disconnect()
