from PIL import Image
from io import BytesIO
import time
Image.MAX_IMAGE_PIXELS = None

def read_img_general(img_path):
    if "s3://" in img_path:
        init_ceph_client_if_needed()
        img_bytes = client.get(img_path)
        image = Image.open(BytesIO(img_bytes)).convert('RGB')
        return image
    else:
        return Image.open(img_path).convert('RGB')

def init_ceph_client_if_needed():
    global client
    if client is None:
        print(f"initializing ceph client ...")
        st = time.time()
        from petrel_client.client import Client  # noqa
        client = Client("~/qlt/petreloss_all.conf")
        ed = time.time()
        print(f"initialize client cost {ed - st:.2f} s")

client = None