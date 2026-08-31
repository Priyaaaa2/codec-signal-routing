"""
generate_foldwise_labels.py

Combines the six out-of-fold score files into one complete, honest oracle
label file covering the full training set (9,537 clips). Each clip's label
comes from the fold model that never saw it during training.
"""

import csv
import numpy as np


def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def fuse_and_label(iframe_path, mv_path, res_path, wi=2.0, wm=1.0, wr=1.0):
    with np.load(iframe_path, allow_pickle=True) as iframe, \
         np.load(mv_path, allow_pickle=True) as mv, \
         np.load(res_path, allow_pickle=True) as residual:

        n = len(mv['names'])
        i_score = np.array([s[0] for s in iframe['scores']])
        mv_score = np.array([s[0] for s in mv['scores']])
        res_score = np.array([s[0] for s in residual['scores']])

        i_label = np.array(iframe['labels'])
        names = mv['names']

        combined_score = i_score * wi + mv_score * wm + res_score * wr
        combined_prob = softmax(combined_score, axis=1)

        predicted = np.argmax(combined_score, axis=1)
        confidence = np.max(combined_prob, axis=1)
        correct = (predicted == i_label).astype(int)

        rows = []
        for i in range(n):
            rows.append({
                'clip_path': names[i],
                'true_label': int(i_label[i]),
                'car_predicted_label': int(predicted[i]),
                'car_correct': int(correct[i]),
                'car_confidence': float(confidence[i]),
            })
        return rows, correct.mean()


scores_dir = 'scores/foldwise'

rows_b, acc_a_on_b = fuse_and_label(
    f'{scores_dir}/foldA_on_foldB_iframe.pkl.npz',
    f'{scores_dir}/foldA_on_foldB_mv.pkl.npz',
    f'{scores_dir}/foldA_on_foldB_residual.pkl.npz',
)
print(f"Fold A models on Fold B clips: {acc_a_on_b:.4f} ({len(rows_b)} clips)")

rows_a, acc_b_on_a = fuse_and_label(
    f'{scores_dir}/foldB_on_foldA_iframe.pkl.npz',
    f'{scores_dir}/foldB_on_foldA_mv.pkl.npz',
    f'{scores_dir}/foldB_on_foldA_residual.pkl.npz',
)
print(f"Fold B models on Fold A clips: {acc_b_on_a:.4f} ({len(rows_a)} clips)")

all_rows = rows_a + rows_b
n_total = len(all_rows)
n_correct = sum(r['car_correct'] for r in all_rows)

print(f"\nTotal clips: {n_total}")
print(f"CAR correct (label 1): {n_correct} ({100*n_correct/n_total:.2f}%)")
print(f"CAR wrong (label 0): {n_total - n_correct} ({100*(n_total-n_correct)/n_total:.2f}%)")

with open('oracle_labels_trainset_foldwise.csv', 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['clip_path', 'true_label', 'car_predicted_label', 'car_correct', 'car_confidence'])
    writer.writeheader()
    writer.writerows(all_rows)

print("\nSaved to oracle_labels_trainset_foldwise.csv")
