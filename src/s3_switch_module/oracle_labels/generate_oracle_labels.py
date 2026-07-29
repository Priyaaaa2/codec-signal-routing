import argparse
import csv
import numpy as np


def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def main():
    parser = argparse.ArgumentParser(description="Generate CAR oracle routing labels")
    parser.add_argument('--iframe', type=str, required=True,
                         help='I-frame stream .npz score file')
    parser.add_argument('--mv', type=str, required=True,
                         help='Motion vector stream .npz score file')
    parser.add_argument('--res', type=str, required=True,
                         help='Residual stream .npz score file')
    parser.add_argument('--wi', type=float, default=2.0,
                         help='I-frame fusion weight (CoViAR default: 2.0)')
    parser.add_argument('--wm', type=float, default=1.0,
                         help='Motion vector fusion weight (CoViAR default: 1.0)')
    parser.add_argument('--wr', type=float, default=1.0,
                         help='Residual fusion weight (CoViAR default: 1.0)')
    parser.add_argument('--output', type=str, required=True,
                         help='Output CSV path for oracle labels')
    args = parser.parse_args()

    with np.load(args.iframe, allow_pickle=True) as iframe, \
         np.load(args.mv, allow_pickle=True) as mv, \
         np.load(args.res, allow_pickle=True) as residual:

        n = len(mv['names'])

        i_score = np.array([s[0] for s in iframe['scores']])
        mv_score = np.array([s[0] for s in mv['scores']])
        res_score = np.array([s[0] for s in residual['scores']])

        i_label = np.array(iframe['labels'])
        mv_label = np.array(mv['labels'])
        res_label = np.array(residual['labels'])

        assert np.alltrue(i_label == mv_label) and np.alltrue(i_label == res_label), \
            "Label mismatch across streams — clip ordering does not match!"

        names = mv['names']

        combined_score = i_score * args.wi + mv_score * args.wm + res_score * args.wr
        combined_prob = softmax(combined_score, axis=1)

        predicted = np.argmax(combined_score, axis=1)
        confidence = np.max(combined_prob, axis=1)
        correct = (predicted == i_label).astype(int)

        overall_acc = correct.mean()
        n_correct = int(correct.sum())
        n_wrong = n - n_correct

        print(f"Total clips: {n}")
        print(f"CAR correct (label 1): {n_correct} ({100*n_correct/n:.2f}%)")
        print(f"CAR wrong (label 0):   {n_wrong} ({100*n_wrong/n:.2f}%)")
        print(f"Overall CAR accuracy on this set: {overall_acc:.4f}")

        with open(args.output, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'clip_path', 'true_label', 'car_predicted_label',
                'car_correct', 'car_confidence'
            ])
            for i in range(n):
                writer.writerow([
                    names[i],
                    int(i_label[i]),
                    int(predicted[i]),
                    int(correct[i]),
                    float(confidence[i]),
                ])

        print(f"Oracle labels written to {args.output}")


if __name__ == '__main__':
    main()