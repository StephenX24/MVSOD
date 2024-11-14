import json
import numpy as np
from collections import defaultdict

data_path = 'eval.json.bak'
with open(data_path, 'r') as f:
    data = json.load(f)


video_ids = data['video_id']
frame_ids = data['frame_id']
loss_gious = data['loss_giou']
record = {}
for video_id, frame_id, loss_giou in zip(video_ids, frame_ids, loss_gious):
    for v_id, f_id, loss in zip(video_id, frame_id, loss_giou):
        if v_id[0] not in record:
            record[v_id[0]] = {'frame_id': [], 'loss_giou': []}
        if f_id[0] not in record[v_id[0]]['frame_id']:
            record[v_id[0]]['frame_id'].append(f_id[0])
            record[v_id[0]]['loss_giou'].append(loss)

videos = []
avg_loss = []
for v_id in record.keys():
    f_ids = record[v_id]['frame_id']
    loss = record[v_id]['loss_giou']
    videos.append(v_id)
    avg_loss.append(np.array(loss).mean())
    print(f"video id: {v_id}, loss: {np.array(loss).mean()}")

avg_loss = np.array(avg_loss)
order_ids = np.argsort(avg_loss)

print("The top N best video id")
for index in order_ids[0:40]:
    print(f"video id: {videos[index]}, loss: {avg_loss[index]}")

