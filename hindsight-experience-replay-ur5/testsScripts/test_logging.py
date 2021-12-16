import logging
import os

log_file_name = "testlog.log"
logging.basicConfig(filename=log_file_name, level="INFO")
logging.info("fucking whore i hate your guts")
logging.info("fucking whore i hate your guts")
logging.info("waiting for print")

if os.path.exists("./testlog.log"):
    print("exists")
else:
    print("no log file created")
