"""
evaluate_switch_classconf_on_testset.py

Final, honest test-set evaluation for the 2-feature (class + confidence
only) switch, using real deployed CAR and SAR models, no fold models,
no training-set numbers mixed in.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

car = pd.read_csv('oracle_labels/oracle_labels_ucf101_test.csv')
i3d = pd.read_csv('/tmp/i3d_test_results.csv')
class_lookup = pd.read_csv('class_fail_rate_lookup.csv', index_col=0)['fail_rate'].to_dict()

df = car.merge(i3d, on='clip_path', how='inner')
assert len(df) == 3783, f"Expected 3783 clips, got {len(df)}"

df['class_name'] = df['clip_path'].apply(lambda x: x.split('/')[0])
df['predicted_class_fail_rate_loo'] = df['class_name'].map(class_lookup)
df['predicted_class_fail_rate_loo'] = df['predicted_class_fail_rate_loo'].fillna(
    df['predicted_class_fail_rate_loo'].mean())

ckpt = torch.load('switch_mlp_classconf_only.pth', map_location='cpu', weights_only=False)
FEATURES = ckpt['features']
scaler_mean = ckpt['scaler_mean']
scaler_scale = ckpt['scaler_scale']
threshold = ckpt['threshold']
print("Features:", FEATURES)

X = df[FEATURES].values.astype(np.float32)
X_scaled = (X - scaler_mean) / scaler_scale


class SwitchMLP(nn.Module):
    def __init__(self, d_in, hidden=32, p_drop=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden), nn.ReLU(), nn.Dropout(p_drop),
            nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Dropout(p_drop),
            nn.Linear(hidden // 2, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


model = SwitchMLP(len(FEATURES))
model.load_state_dict(ckpt['state_dict'])
model.eval()

with torch.no_grad():
    probs = torch.sigmoid(model(torch.from_numpy(X_scaled.astype(np.float32)))).numpy()

route_to_sar = probs >= threshold
car_correct = df['car_correct'].values.astype(bool)
i3d_correct = df['i3d_correct'].values.astype(bool)

n = len(df)
n_to_sar = route_to_sar.sum()
n_to_car = n - n_to_sar

final_correct = np.where(route_to_sar, i3d_correct, car_correct)
system_accuracy = final_correct.mean()

car_acc_when_kept = car_correct[~route_to_sar].mean() if n_to_car > 0 else float('nan')
i3d_acc_when_routed = i3d_correct[route_to_sar].mean() if n_to_sar > 0 else float('nan')

always_car_acc = car_correct.mean()
always_sar_acc = i3d_correct.mean()

n_car_failures = (~car_correct).sum()
caught = ((~car_correct) & route_to_sar).sum()
catch_rate = caught / n_car_failures if n_car_failures > 0 else float('nan')

print("=" * 60)
print("FINAL TEST-SET EVAL — CLASS+CONFIDENCE ONLY (2 features)")
print("=" * 60)
print(f"Always-CAR baseline: {always_car_acc*100:.2f}%")
print(f"Always-SAR baseline: {always_sar_acc*100:.2f}%")
print()
print(f"Clips routed to CAR: {n_to_car} ({100*n_to_car/n:.1f}%)")
print(f"Clips routed to SAR: {n_to_sar} ({100*n_to_sar/n:.1f}%)")
print()
print(f"CAR accuracy on clips it kept: {car_acc_when_kept*100:.2f}%")
print(f"SAR accuracy on clips routed to it: {i3d_acc_when_routed*100:.2f}%")
print()
print(f"Router catch rate: {catch_rate*100:.2f}%")
print()
print(f"FINAL SYSTEM ACCURACY: {system_accuracy*100:.2f}%")
print()
print("Compare against full 8-feature version: 94.32% (24.8% to SAR)")
