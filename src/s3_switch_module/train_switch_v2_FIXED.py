import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

FEATURES = ['predicted_class_fail_rate_loo', 'car_confidence',
            'mv_mean', 'mv_std', 'mv_max', 'mv_sparsity', 'residual_energy_mean', 'num_frames']

df = pd.read_csv('h264_switch_training_data_v2_FIXED.csv')
FEATURES = [f for f in FEATURES if f in df.columns]
print("Using features:", FEATURES)

X = df[FEATURES].values.astype(np.float32)
y = (1 - df['car_correct'].values).astype(np.float32)

X_dev, X_eval, y_dev, y_eval = train_test_split(X, y, test_size=0.30, stratify=y, random_state=SEED)
X_tr, X_val, y_tr, y_val = train_test_split(X_dev, y_dev, test_size=0.20, stratify=y_dev, random_state=SEED)

scaler = StandardScaler().fit(X_tr)
X_tr, X_val, X_eval = map(scaler.transform, (X_tr, X_val, X_eval))
t = lambda a: torch.from_numpy(np.asarray(a, dtype=np.float32))
X_tr_t, y_tr_t = t(X_tr), t(y_tr)
X_val_t, y_val_t = t(X_val), t(y_val)
X_eval_t, y_eval_t = t(X_eval), t(y_eval)

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
n_pos = y_tr.sum(); n_neg = len(y_tr) - n_pos
pos_weight = torch.tensor(n_neg / n_pos)
print(f"pos_weight = {pos_weight.item():.2f}")

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimiser = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_tr_t, y_tr_t), batch_size=64, shuffle=True)

def balanced_accuracy(probs, targets, thr):
    pred = (probs >= thr).float()
    pos, neg = targets == 1, targets == 0
    rec_pos = (pred[pos] == 1).float().mean().item() if pos.any() else 0.0
    rec_neg = (pred[neg] == 0).float().mean().item() if neg.any() else 0.0
    return 0.5 * (rec_pos + rec_neg), rec_pos, rec_neg

best_val, best_state, patience, since_best = -1.0, None, 30, 0
for epoch in range(1, 301):
    model.train()
    for xb, yb in loader:
        optimiser.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        optimiser.step()
    model.eval()
    with torch.no_grad():
        val_probs = torch.sigmoid(model(X_val_t))
    val_bal, _, _ = balanced_accuracy(val_probs, y_val_t, 0.5)
    if val_bal > best_val:
        best_val, since_best = val_bal, 0
        best_state = {k: v.clone() for k, v in model.state_dict().items()}
    else:
        since_best += 1
    if epoch % 25 == 0:
        print(f"epoch {epoch:3d}  val bal-acc {val_bal:.4f}")
    if since_best >= patience:
        print(f"early stop at epoch {epoch}")
        break

model.load_state_dict(best_state)
print(f"\nbest val bal-acc: {best_val:.4f}")

with torch.no_grad():
    val_probs = torch.sigmoid(model(X_val_t))
best_thr, best_thr_score = 0.5, -1.0
for thr in np.arange(0.05, 0.96, 0.05):
    score, _, _ = balanced_accuracy(val_probs, y_val_t, thr)
    if score > best_thr_score:
        best_thr_score, best_thr = score, thr
print(f"selected threshold: {best_thr:.2f}")

with torch.no_grad():
    eval_probs = torch.sigmoid(model(X_eval_t))
bal, rec_pos, rec_neg = balanced_accuracy(eval_probs, y_eval_t, best_thr)
frac_sar = (eval_probs >= best_thr).float().mean().item()
print(f"Held-out eval: bal-acc={bal:.4f} catch-SAR={rec_pos:.4f} keep-CAR={rec_neg:.4f} pct-to-SAR={100*frac_sar:.1f}%")

torch.save({'state_dict': model.state_dict(), 'scaler_mean': scaler.mean_,
            'scaler_scale': scaler.scale_, 'features': FEATURES, 'threshold': float(best_thr)},
           'switch_mlp_v2_FIXED.pth')
print("saved switch_mlp_v2_FIXED.pth")
