import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('oracle_labels_ucf101_test.csv')

train_df, eval_df = train_test_split(
    df, test_size=0.3, stratify=df['car_correct'], random_state=42
)

train_df.to_csv('oracle_labels_train_split.csv', index=False)
eval_df.to_csv('oracle_labels_eval_split.csv', index=False)

print(f"Train: {len(train_df)} clips, {train_df['car_correct'].sum()} correct, {(train_df['car_correct']==0).sum()} wrong")
print(f"Eval:  {len(eval_df)} clips, {eval_df['car_correct'].sum()} correct, {(eval_df['car_correct']==0).sum()} wrong")
