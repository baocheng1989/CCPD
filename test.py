import torch
from models.experimental import attempt_load

model=attempt_load("weights/best_yolov5s_final.pt")
# print(model.state_dict())