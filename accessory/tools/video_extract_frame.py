import json
import time
import decord
from io import BytesIO
from PIL import Image
from pathlib import Path
from multiprocessing import Pool


def extract_video(l_video_ids, rank):

    print(f"rank {rank} initializing ceph client ...")
    st = time.time()
    from petrel_client.client import Client  # noqa

    client = Client("~/qlt/petreloss_all.conf")
    ed = time.time()
    print(f"rank {rank} initialize client cost {ed - st:.2f} s")

    for idx, video_id in enumerate(l_video_ids):
        num_frames = 10

        file_path = f'cluster_ssd_share:s3://video_pub/ANet_320p_fps30/train/{video_id}.mp4'

        l = list(client.get_file_iterator(file_path))
        if len(l) == 0:
            file_path = f'cluster_ssd_share:s3://video_pub/ANet_320p_fps30/val/{video_id}.mp4'
            l = list(client.get_file_iterator(file_path))
        assert len(l) == 1

        Path(f"frames_temp/{video_id}").mkdir(parents=True, exist_ok=True)

        bytes = client.get(file_path)
        video_reader = decord.VideoReader(BytesIO(bytes))
        start, end = 0, len(video_reader)

        frame_ids = [round((end - start - 1) / (num_frames - 1) * i + start) for i in range(num_frames)]
        frames = video_reader.get_batch(frame_ids).asnumpy()

        frames = [Image.fromarray(frames[i]).convert('RGB') for i in range(frames.shape[0])]

        for image, frame_id in zip(frames, frame_ids):
            image.save(f"frames_temp/{video_id}/{frame_id}.jpg")

        if idx % 100 == 0:
            print(f"rank {rank} ({idx}/{len(l_video_ids)}) done")


if __name__=="__main__":
    with open("../data/Video-ChatGPT/VideoInstruct_Dataset.json", 'r') as f:
        contents = json.load(f)

    all_video_ids = list(set([_['video_id'] for _ in contents]))

    workers = 8
    pool = Pool(workers)
    for worker_id in range(workers):
        pool.apply_async(extract_video, [all_video_ids[worker_id::workers], worker_id])
    # debug
    # extract_video(all_video_ids[0::workers], 0)

    pool.close()
    pool.join()