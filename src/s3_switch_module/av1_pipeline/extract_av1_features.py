import argparse
import csv
import json
import os
import subprocess
import tempfile
import numpy as np

INSPECT_BIN = '/mnt/faster3/pmk46/aom_build/examples/inspect'
FFMPEG_BIN = '/usr/bin/ffmpeg'

# Approximate pixel size per blockSize code (width, height), per AV1 spec
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


def run_inspect(ivf_path, num_frames=10):
    result = subprocess.run(
        [INSPECT_BIN, ivf_path, '-mv', '-bs', f'--limit={num_frames}'],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr[-300:])
    return json.loads(result.stdout)


def extract_features_for_clip(mp4_path, num_frames=10):
    with tempfile.TemporaryDirectory() as tmpdir:
        ivf_path = os.path.join(tmpdir, 'stream.ivf')
        extract_av1_stream(mp4_path, ivf_path)
        frames = run_inspect(ivf_path, num_frames)

    mv_magnitudes = []
    block_area_px = []  # area of each coded block, weighted per-block (not per-pixel)

    for frame in frames:
        if frame is None:
            continue
        if frame.get('frameType') == 0:  # keyframe, no motion
            continue

        mv_grid = frame.get('motionVectors', [])
        for row in mv_grid:
            for cell in row:
                # cell = [mv0_row, mv0_col, mv1_row, mv1_col] (1/8-pel units)
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
        # Fraction of blocks smaller than 16x16 (256px): fine partitioning
        # indicates complex/detailed motion the encoder needed to model precisely
        'av1_fine_partition_frac': float(np.mean(all_area < 256)),
    }
    return features


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--clip-list', type=str, required=True)
    parser.add_argument('--data-root', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--num-frames', type=int, default=10)
    args = parser.parse_args()

    with open(args.clip_list) as f:
        clips = [line.strip().split()[0] for line in f if line.strip()]

    rows = []
    n_failed = 0
    for i, rel_path in enumerate(clips):
        mp4_path = os.path.join(args.data_root, rel_path[:-4] + '.mp4')
        try:
            feats = extract_features_for_clip(mp4_path, args.num_frames)
        except Exception as e:
            print(f"[{i}] FAILED {rel_path}: {e}")
            feats = None

        if feats is None:
            n_failed += 1
            continue

        feats['clip_path'] = rel_path
        rows.append(feats)

        if (i + 1) % 200 == 0:
            print(f"{i+1}/{len(clips)} done ({n_failed} failed)")

    fieldnames = list(rows[0].keys()) if rows else []
    with open(args.output, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {args.output} ({n_failed} failed)")


if __name__ == '__main__':
    main()
