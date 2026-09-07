"""
evaluate_switch_v2v3_FIXED_on_testset.py

Final, honest, LEAK-FREE test-set evaluation for v2 (class+conf+H.264) and
v3 (class+conf+H.264+AV1). Uses CAR's PREDICTED class throughout.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import json

car = pd.read_csv('oracle_labels/oracle_labels_ucf101_test.csv')
i3d = pd.read_csv('i3d_test_results.csv')
h264_feat = pd.read_csv('features/all_features_v2.csv')
av1_feat = pd.read_csv('av1_pipeline/av1_features_v2_all.csv')

with open('class_fail_rate_lookup_corrected.json') as f:
    class_lookup = json.load(f)
idx_to_class = {}
with open('classInd.txt') as f:
    for line in f:
        idx, name = line.strip().split()
        idx_to_class[int(idx) - 1] = name

df = car.merge(i3d, on='clip_path', how='inner').merge(h264_feat, on='clip_path', how='inner').merge(av1_feat, on='clip_path', how='inner')
assert len(df) == 3783, f"Expected 3783 clips, got {len(df)}"

df['predicted_class_name'] = df['car_predicted_label'].map(idx_to_class)
df['predicted_class_fail_rate_loo'] = df['predicted_class_name'].map(class_lookup)
train_mean = sum(class_lookup.values()) / len(class_lookup)
df['predicted_class_fail_rate_loo'] = df['predicted_class_fail_rate_loo'].fillna(train_mean)

car_correct = df['car_correct'].values.astype(bool)
i3d_correct = df['i3d_correct'].values.astype(bool)
always_car_acc = car_correct.mean()
always_sar_acc = i3d_correct.mean()
n = len(df)
n_car_failures = (~car_correct).sum()


class SwitchMLP(nn.Module):
    def __init__(self, d_in, hidden=32, p_drop=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden), nn.ReLU(), nn.Dropout(p_drop),
            nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Dropout(p_drop),
            nn.Linear(hidden // 2, 1))
    def forward(self, x):
        return self.net(x).squeeze(-1)


def evaluate(checkpoint_path, label):
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    FEATURES = ckpt['features']
    X = df[FEATURES].values.astype(np.float32)
    X_scaled = (X - ckpt['scaler_mean']) / ckpt['scaler_scale']

    model = SwitchMLP(len(FEATURES))
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model(torch.from_numpy(X_scaled.astype(np.float32)))).numpy()

    route_to_sar = probs >= ckpt['threshold']
    n_to_sar = route_to_sar.sum()
    n_to_car = n - n_to_sar
    final_correct = np.where(route_to_sar, i3d_correct, car_correct)
    system_accuracy = final_correct.mean()
    car_acc_when_kept = car_correct[~route_to_sar].mean() if n_to_car > 0 else float('nan')
    i3d_acc_when_routed = i3d_correct[route_to_sar].mean() if n_to_sar > 0 else float('nan')
    caught = ((~car_correct) & route_to_sar).sum()
    catch_rate = caught / n_car_failures

    print(f"\n=== {label} ===")
    print(f"Features: {FEATURES}")
    print(f"Clips routed to SAR: {n_to_sar} ({100*n_to_sar/n:.1f}%)")
    print(f"CAR accuracy on retained clips: {car_acc_when_kept*100:.2f}%")
    print(f"SAR accuracy on routed clips: {i3d_acc_when_routed*100:.2f}%")
    print(f"Router catch rate: {catch_rate*100:.2f}%")
    print(f"FINAL SYSTEM ACCURACY: {system_accuracy*100:.2f}%")


print(f"Always-CAR: {always_car_acc*100:.2f}%")
print(f"Always-SAR: {always_sar_acc*100:.2f}%")

evaluate('switch_mlp_v2_FIXED.pth', 'v2: class+conf+H.264 (FIXED)')
evaluate('switch_mlp_v3_FIXED.pth', 'v3: class+conf+H.264+AV1 (FIXED)')
