import argparse
import csv
import json
import os
import subprocess
import tempfile
import numpy as np

INSPECT_BIN = '/mnt/faster3/pmk46/aom_build/examples/inspect'
FFMPEG_BIN = '/usr/bin/ffmpeg'

BLOCK_SIZE_PIXELS = {
    0: (4, 4), 1: (4, 8), 2: (8, 4), 3: (8, 8), 4: (8, 16), 5: (16, 8),
    6: (16, 16), 7: (16, 32), 8: (32, 16), 9: (32, 32), 10: (32, 64),
    11: (64, 32), 12: (64, 64), 13: (64, 128), 14: (128, 64), 15: (128, 128),
    16: (4, 16), 17: (16, 4), 18: (8, 32), 19: (32, 8), 20: (16, 64), 21: (64, 16),
}


def extract_av1_stream(mp4_path, ivf_path):
    subprocess.run(
        [FFMPEG_BIN, '-y', '-loglevel', 'error', '-i', mp4_path,
         '-c:v', 'copy', '-f', 'ivf', ivf_path],
        check=True
    )


def run_inspect_full(ivf_path):
    # No --limit: decode the ENTIRE clip so we can sample across its full
    # duration, matching the H.264 extractor's methodology.
    result = subprocess.run(
        [INSPECT_BIN, ivf_path, '-mv', '-bs'],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr[-300:])
    return json.loads(result.stdout)


def extract_features_for_clip(mp4_path, num_segments=5):
    with tempfile.TemporaryDirectory() as tmpdir:
        ivf_path = os.path.join(tmpdir, 'stream.ivf')
        extract_av1_stream(mp4_path, ivf_path)
        all_frames = run_inspect_full(ivf_path)

    # Keep only usable inter-predicted frames (skip keyframes and None entries)
    usable = [f for f in all_frames if f is not None and f.get('frameType') != 0]

    if len(usable) == 0:
        return None

    # Sample num_segments points spread across the usable frames,
    # same logic as the H.264 extractor's seg_size sampling.
    n = len(usable)
    seg_size = max(1, n // num_segments)
    sample_indices = [min(seg * seg_size, n - 1) for seg in range(num_segments)]
    sampled_frames = [usable[i] for i in sample_indices]

    mv_magnitudes = []
    block_area_px = []

    for frame in sampled_frames:
        mv_grid = frame.get('motionVectors', [])
        for row in mv_grid:
            for cell in row:
                mag0 = np.sqrt(cell[0] ** 2 + cell[1] ** 2) / 8.0
                mv_magnitudes.append(mag0)

        bs_map = frame.get('blockSize', [])
        for row in bs_map:
            for code in row:
                w, h = BLOCK_SIZE_PIXELS.get(code, (8, 8))
                block_area_px.append(w * h)

    if len(mv_magnitudes) == 0:
        return None

    all_mv = np.array(mv_magnitudes)
    all_area = np.array(block_area_px) if block_area_px else np.array([64.0])

    features = {
        'av1_mv_mean': float(np.mean(all_mv)),
        'av1_mv_std': float(np.std(all_mv)),
        'av1_mv_max': float(np.max(all_mv)),
        'av1_mv_sparsity': float(np.mean(all_mv < 1.0)),
        'av1_block_area_mean': float(np.mean(all_area)),
        'av1_block_area_std': float(np.std(all_area)),
        'av1_fine_partition_frac': float(np.mean(all_area < 256)),
        'av1_total_frames': int(len(all_frames)),
        'av1_usable_frames': int(n),
    }
    return features


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--clip-list', type=str, required=True)
    parser.add_argument('--data-root', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--num-segments', type=int, default=5)
    args = parser.parse_args()

    with open(args.clip_list) as f:
        clips = [line.strip().split()[0] for line in f if line.strip()]

    rows = []
    n_failed = 0
    for i, rel_path in enumerate(clips):
        mp4_path = os.path.join(args.data_root, rel_path[:-4] + '.mp4')
        try:
            feats = extract_features_for_clip(mp4_path, args.num_segments)
        except Exception as e:
            print(f"[{i}] FAILED {rel_path}: {e}")
            feats = None

        if feats is None:
            n_failed += 1
            continue

        feats['clip_path'] = rel_path
        rows.append(feats)

        if (i + 1) % 100 == 0:
            print(f"{i+1}/{len(clips)} done ({n_failed} failed)")

    fieldnames = list(rows[0].keys()) if rows else []
    with open(args.output, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {args.output} ({n_failed} failed)")


if __name__ == '__main__':
    main()
