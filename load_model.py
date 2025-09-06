import json, torch
from robust_sleep_net.modulo_net import ModuloNet  # مسیر ماژول را با نصب/کلون خودت تنظیم کن

DIR = "./pretrained_model/0dfcee73-055a-4c4d-929c-8fdf630e14f1/best_model"  # پوشه‌ای که 5 فایل داخلش است

with open(f"{DIR}/groups.json") as f: groups = json.load(f)
with open(f"{DIR}/features.json") as f: features = json.load(f)
with open(f"{DIR}/normalization_parameters.json") as f: norm = json.load(f)
with open(f"{DIR}/net_params.json") as f: net_params = json.load(f)

net = ModuloNet(groups, features, norm, net_params)             # معماری + نرمال‌سازی
state = torch.load(f"{DIR}/state.torch", map_location="cpu")     # وزن‌ها
net.load_state_dict(state)                                       # لود وزن‌ها
net.eval()
