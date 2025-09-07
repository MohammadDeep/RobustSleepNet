import json, torch
from robust_sleep_net.models.modulo_net import ModuloNet  # مسیر ماژول را با نصب/کلون خودت تنظیم کن

DIR = "./pretrained_model/fdbd193a-0408-4fc8-9a6a-a18351459512/best_model"  # پوشه‌ای که 5 فایل داخلش است

with open(f"{DIR}/groups.json") as f: groups = json.load(f)
with open(f"{DIR}/features.json") as f: features = json.load(f)
with open(f"{DIR}/normalization_parameters.json") as f: norm = json.load(f)
with open(f"{DIR}/net_params.json") as f: net_params = json.load(f)

net = ModuloNet(groups, features, norm, net_params)             # معماری + نرمال‌سازی
state = torch.load(f"{DIR}/state.torch", map_location="cpu")     # وزن‌ها
net.load_state_dict(state)                                       # لود وزن‌ها
net.eval()


from torch import nn
from pprint import pprint

def print_model_info(net):
    print("\n=== Model Summary ===")
    print("Device:", net.device)
    print("n_class:", net.net_parameters.get("n_class"))
    print("output_mode:", getattr(net, "output_mode", None),
          "| eval_output_mode:", getattr(net, "eval_output_mode", None))
    print("input_temporal_context:", net.net_parameters.get("input_temporal_context"))

    # Groups / Encoders / Reducers
    print("\n--- Groups / Encoders / Reducers ---")
    encs = net.net_parameters.get("encoders", {})
    reds = net.net_parameters.get("reducers", {})
    for g in net.groups:
        print(f"\nGroup: {g}")
        print("  raw shape:", net.groups[g].get("shape"))
        print("  encoder:", encs[g]["type"], "args=", encs[g].get("args"))
        print("  encoder_input_shape:", net.groups[g].get("encoder_input_shape"))
        print("  reducer:", reds[g]["type"], "args=", reds[g].get("args"))
        print("  reducer_input_shape:", net.groups[g].get("reducer_input_shape"))

    # Features encoder (اگر هست)
    print("\n--- Features encoder ---")
    if getattr(net, "features_encoder", None) is not None:
        fe = net.net_parameters.get("features_encoder", {})
        print("  type:", fe.get("type"), "args=", fe.get("args"))
        print("  out_features:", getattr(net.features_encoder, "out_features", None))
        # نام فیچرها
        print("  feature names:",
              list(net.features.keys()) if isinstance(net.features, dict) else net.features)
    else:
        print("  (no features_encoder)")

    # Sequence encoder + Classifier
    print("\n--- Sequence / Classifier ---")
    print("  sequence_encoder:", net.sequence_encoder.__class__.__name__)
    print("  sequence_encoder.output_size:", getattr(net.sequence_encoder, "output_size", None))
    if isinstance(net.classifier, nn.Linear):
        print(f"  classifier: Linear({net.classifier.in_features} -> {net.classifier.out_features})")
    else:
        print("  classifier:", net.classifier)

    # Normalization pipeline (خلاصه)
    print("\n--- Normalization (signals) ---")
    for i, grp in enumerate(net.normalization_parameters["signals"]):
        ops = [op["type"] for op in grp["normalization"]]
        print(f"  signals[{i}] ops:", ops)
    print("--- Normalization (features) ---")
    for i, grp in enumerate(net.normalization_parameters["features"]):
        ops = [op["type"] for op in grp["normalization"]]
        print(f"  features[{i}] ops:", ops)

    # Params
    total = sum(p.numel() for p in net.parameters())
    trainable = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"\nParameters: total={total:,} | trainable={trainable:,}")

    # ماژول‌های سطح بالا
    print("\n--- Top-level modules ---")
    for name, _ in net.named_children():
        print(" ", name)

    # اگر خواستی کلیدهای state_dict و ابعاد‌شان را ببینی (کامنت را بردار)
    # print("\n--- state_dict keys (sample) ---")
    # for i, (k, v) in enumerate(net.state_dict().items()):
    #     print(k, tuple(v.shape))
    #     if i > 40: break

print_model_info(net)


import torch
from torch import nn

def _build_dummy_signals(net):
    # ورودی ساختگی با شکل خام (B=1, T=input_temporal_context, C, L)
    sig = {}
    for g, desc in net.groups.items():
        L, C = desc["shape"][0], desc["shape"][1]
        x = torch.zeros((1, net.input_temporal_context, C, L), device=net.device)
        sig[g] = x
    return sig

def print_summary(net):
    dummy_signals = _build_dummy_signals(net)
    rows, hooks = [], []

    def register(name, module):
        # فقط برای ماژول‌های برگ (بدون بچه)
        if any(True for _ in module.children()):
            return
        def hook(mod, inp, out):
            params = sum(p.numel() for p in mod.parameters())
            trainable = sum(p.numel() for p in mod.parameters() if p.requires_grad)
            if isinstance(out, torch.Tensor):
                out_shape = list(out.shape)
            elif isinstance(out, (list, tuple)) and len(out) and isinstance(out[0], torch.Tensor):
                out_shape = [list(o.shape) for o in out]
            else:
                out_shape = str(type(out))
            rows.append((name, mod.__class__.__name__, params, trainable, out_shape))
        hooks.append(module.register_forward_hook(hook))

    for name, module in net.named_modules():
        register(name, module)

    with torch.no_grad():
        net({"signals": dummy_signals, "features": {}})  # forward برای گرفتن خروجی لایه‌ها

    for h in hooks:
        h.remove()

    # چاپ جدول
    print("\n{:<48} {:<28} {:>12} {:>12} {:>20}".format("Layer", "Type", "Params", "Trainable", "Output"))
    print("-"*130)
    for name, cls, p, t, shape in rows:
        print("{:<48} {:<28} {:>12,} {:>12,} {:>20}".format(name, cls, p, t, str(shape)))
    tot = sum(p for _,_,p,_,_ in rows)
    trn = sum(t for _,_,_,t,_ in rows)
    print("-"*130)
    print(f"Total params: {tot:,} | Trainable: {trn:,}")

# فراخوانی:
print_summary(net)

