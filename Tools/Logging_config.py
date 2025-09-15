import os
import logging
from datetime import datetime


def setup_logging(log_directory, log_name="application"):
    if not os.path.exists(log_directory):
        os.makedirs(log_directory)

    file_name = f"{log_name}_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    log_filename = os.path.join(log_directory, file_name)
    logging.basicConfig(
        filename=log_filename,
        filemode='w',
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.DEBUG
    )

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(levelname)s - %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

    logging.info("Logging initialized")

