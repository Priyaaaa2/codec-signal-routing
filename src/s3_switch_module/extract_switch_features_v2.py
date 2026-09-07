import argparse
import csv
import numpy as np
from coviar import get_num_frames, load

GOP_SIZE = 12
GRID = 3  # 3x3 = 9 spatial regions; keep small, this is meant to stay cheap


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


def grid_pool(arr2d, grid=GRID, reduce_fn=np.mean):
    """Pool a 2D array into a grid x grid set of region summaries."""
    h, w = arr2d.shape[:2]
    h_bins = np.array_split(np.arange(h), grid)
    w_bins = np.array_split(np.arange(w), grid)
    out = np.zeros((grid, grid), dtype=np.float32)
    for i, hb in enumerate(h_bins):
        if len(hb) == 0:
            continue
        for j, wb in enumerate(w_bins):
            if len(wb) == 0:
                continue
            region = arr2d[hb[0]:hb[-1] + 1, wb[0]:wb[-1] + 1]
            out[i, j] = reduce_fn(region) if region.size > 0 else 0.0
    return out


def extract_features_for_clip(video_path, num_segments=5):
    num_frames = get_num_frames(video_path)
    if num_frames is None or num_frames < 2:
        return None

    mv_grid_sum = np.zeros((GRID, GRID), dtype=np.float64)
    res_grid_sum = np.zeros((GRID, GRID), dtype=np.float64)
    n_valid = 0

    all_mv_flat = []
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
            all_mv_flat.append(mag)
            mv_grid_sum += grid_pool(mag, GRID, np.mean)
            n_valid += 1

        if res is not None:
            res_energy_map = np.mean(res.astype(np.float32) ** 2, axis=-1) \
                if res.ndim == 3 else res.astype(np.float32) ** 2
            residual_energies.append(np.mean(res_energy_map))
            res_grid_sum += grid_pool(res_energy_map, GRID, np.mean)

    if n_valid == 0 or len(all_mv_flat) == 0:
        return None

    all_mv = np.concatenate([m.flatten() for m in all_mv_flat])
    mv_grid_avg = (mv_grid_sum / n_valid).flatten()
    res_grid_avg = (res_grid_sum / max(1, len(residual_energies))).flatten()

    # Spatial concentration: ratio of max region to mean region.
    # High ratio -> motion concentrated in one area; low ratio -> spread evenly.
    mv_concentration = float(mv_grid_avg.max() / (mv_grid_avg.mean() + 1e-6))
    res_concentration = float(res_grid_avg.max() / (res_grid_avg.mean() + 1e-6))

    features = {
        'mv_mean': float(np.mean(all_mv)),
        'mv_std': float(np.std(all_mv)),
        'mv_max': float(np.max(all_mv)),
        'mv_sparsity': float(np.mean(all_mv < 1.0)),
        'residual_energy_mean': float(np.mean(residual_energies)) if residual_energies else 0.0,
        'num_frames': int(num_frames),
        'mv_concentration': mv_concentration,
        'res_concentration': res_concentration,
    }
    for k in range(GRID * GRID):
        features[f'mv_grid_{k}'] = float(mv_grid_avg[k])
    for k in range(GRID * GRID):
        features[f'res_grid_{k}'] = float(res_grid_avg[k])

    return features


def main():
    parser = argparse.ArgumentParser()
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

    fieldnames = list(rows[0].keys()) if rows else []
    with open(args.output, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} feature rows to {args.output} ({n_failed} failed)")


if __name__ == '__main__':
    main()
