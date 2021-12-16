import datetime
import logging

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M',
                    filename='tmp.log',
                    filemode='a')

logging.info("Welcome to the script.")
logging.warning("this is a warning")
