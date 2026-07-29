import pandas as pd

features = pd.read_csv('features/all_features.csv')
labels = pd.read_csv('oracle_labels/oracle_labels_ucf101_test.csv')

merged = features.merge(labels, on='clip_path', how='inner')

print(f"Features: {len(features)} rows")
print(f"Labels: {len(labels)} rows")
print(f"Merged: {len(merged)} rows")

if len(merged) != len(labels):
    print("WARNING: row count mismatch — some clips did not match between files!")

merged.to_csv('switch_training_data.csv', index=False)
print("Saved to switch_training_data.csv")
