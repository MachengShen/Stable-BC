import numpy as np
import json

def process_demos(cfg):
    data = np.array(json.load(open('data/user_{}/demo.json'.format(cfg.user), 'r')))
    dataset = []
    data = data.tolist()

    for idx in range(len(data)):
        robot_pos = data[idx][:2]
        puck_pos = data[idx][2:4]
        robot_vel = data[idx][4:6]
        if idx > 0:
            puck_prev_pos = data[idx-1][2:4]
        
            dataset.append(robot_pos + puck_pos + puck_prev_pos + robot_vel)

    json.dump(dataset, open('data/user_{}/demo_processed.json'.format(cfg.user), 'w'))
