from io import BytesIO
import logging
import time
from typing import Union

from PIL import Image

Image.MAX_IMAGE_PIXELS = None
logger = logging.getLogger(__name__)

def read_general(path) -> Union[str, BytesIO]:
    if "s3://" in path:
        init_ceph_client_if_needed()
        file_bytes = BytesIO(client.get(path))
        return file_bytes
    else:
        return path

def init_ceph_client_if_needed():
    global client
    if client is None:
        logger.info(f"initializing ceph client ...")
        st = time.time()
        from petrel_client.client import Client  # noqa
        client = Client("../petreloss.conf")
        ed = time.time()
        logger.info(f"initialize client cost {ed - st:.2f} s")

client = None
