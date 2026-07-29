import argparse
import csv
import numpy as np
from coviar import get_num_frames, load

GOP_SIZE = 12


def get_gop_pos(frame_idx, representation):
    gop_index = frame_idx // GOP_SIZE
    gop_pos = frame_idx % GOP_SIZE
    if representation in ['residual', 'mv']:
        if gop_pos == 0:
            gop_index -= 1
            gop_pos = GOP_SIZE - 1
    else:
        gop_pos = 0
    return gop_index, gop_pos


def extract_features_for_clip(video_path, num_segments=5):
    num_frames = get_num_frames(video_path)
    if num_frames is None or num_frames < 2:
        return None

    mv_magnitudes = []
    residual_energies = []

    seg_size = max(1, (num_frames - 1) // num_segments)

    for seg in range(num_segments):
        frame_idx = min(seg * seg_size + 1, num_frames - 1)

        mv_gop_idx, mv_gop_pos = get_gop_pos(frame_idx, 'mv')
        res_gop_idx, res_gop_pos = get_gop_pos(frame_idx, 'residual')

        mv = load(video_path, mv_gop_idx, mv_gop_pos, 1, True)
        res = load(video_path, res_gop_idx, res_gop_pos, 2, True)

        if mv is not None:
            mag = np.sqrt(mv[..., 0].astype(np.float32) ** 2 +
                          mv[..., 1].astype(np.float32) ** 2)
            mv_magnitudes.append(mag)

        if res is not None:
            residual_energies.append(np.mean(res.astype(np.float32) ** 2))

    if len(mv_magnitudes) == 0:
        return None

    all_mv = np.concatenate([m.flatten() for m in mv_magnitudes])

    features = {
        'mv_mean': float(np.mean(all_mv)),
        'mv_std': float(np.std(all_mv)),
        'mv_max': float(np.max(all_mv)),
        'mv_sparsity': float(np.mean(all_mv < 1.0)),
        'residual_energy_mean': float(np.mean(residual_energies)) if residual_energies else 0.0,
        'num_frames': int(num_frames),
    }
    return features


def main():
    parser = argparse.ArgumentParser(description="Extract codec-level features for switch MLP")
    parser.add_argument('--clip-list', type=str, required=True)
    parser.add_argument('--data-root', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--num-segments', type=int, default=5)
    args = parser.parse_args()

    with open(args.clip_list) as f:
        clips = [line.strip().split() for line in f if line.strip()]

    rows = []
    n_failed = 0
    for i, parts in enumerate(clips):
        rel_path = parts[0]
        video_path = args.data_root.rstrip('/') + '/' + rel_path[:-4] + '.mp4'

        try:
            feats = extract_features_for_clip(video_path, args.num_segments)
        except Exception as e:
            print(f"[{i}] FAILED on {rel_path}: {e}")
            feats = None

        if feats is None:
            n_failed += 1
            continue

        feats['clip_path'] = rel_path
        rows.append(feats)

        if (i + 1) % 200 == 0:
            print(f"{i+1}/{len(clips)} clips processed")

    fieldnames = ['clip_path', 'mv_mean', 'mv_std', 'mv_max',
                  'mv_sparsity', 'residual_energy_mean', 'num_frames']
    with open(args.output, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} feature rows to {args.output} ({n_failed} clips failed)")


if __name__ == '__main__':
    main()