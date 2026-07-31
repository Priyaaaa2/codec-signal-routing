import argparse
import os
import subprocess
import json


def get_clip_bitrate_kbps(path):
    result = subprocess.run(
        ['ffprobe', '-v', 'error', '-count_frames', '-select_streams', 'v:0',
         '-show_entries', 'stream=nb_read_frames,r_frame_rate',
         '-of', 'json', path],
        capture_output=True, text=True
    )
    info = json.loads(result.stdout)['streams'][0]
    nframes = int(info['nb_read_frames'])
    num, den = info['r_frame_rate'].split('/')
    fps = float(num) / float(den)
    duration = nframes / fps
    size_bits = os.path.getsize(path) * 8
    return size_bits / duration / 1000  # kbps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--clip-list', type=str, required=True)
    parser.add_argument('--h264-root', type=str, required=True)
    parser.add_argument('--av1-root', type=str, required=True)
    parser.add_argument('--encoder', type=str, default='libsvtav1')
    parser.add_argument('--preset', type=str, default='8')
    args = parser.parse_args()

    with open(args.clip_list) as f:
        clips = [line.strip().split()[0] for line in f if line.strip()]

    n_ok, n_failed = 0, 0
    for i, rel_path in enumerate(clips):
        h264_path = os.path.join(args.h264_root, rel_path[:-4] + '.mp4')
        av1_path = os.path.join(args.av1_root, rel_path[:-4] + '.mp4')
        os.makedirs(os.path.dirname(av1_path), exist_ok=True)

        if os.path.exists(av1_path):
            n_ok += 1
            continue

        try:
            kbps = get_clip_bitrate_kbps(h264_path)
        except Exception as e:
            print(f"[{i}] BITRATE CHECK FAILED {rel_path}: {e}")
            n_failed += 1
            continue

        target_kbps = max(50, kbps)  # floor to avoid degenerate near-0 targets

        cmd = [
            '/usr/bin/ffmpeg', '-y', '-loglevel', 'error',
            '-i', h264_path,
            '-c:v', args.encoder,
            '-b:v', f'{target_kbps:.0f}k',
            '-svtav1-params', f'preset={args.preset}',
            av1_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"[{i}] ENCODE FAILED {rel_path}: {result.stderr[-300:]}")
            n_failed += 1
            continue

        n_ok += 1
        if (i + 1) % 200 == 0:
            print(f"{i+1}/{len(clips)} done ({n_ok} ok, {n_failed} failed)")

    print(f"Finished: {n_ok} ok, {n_failed} failed out of {len(clips)}")


if __name__ == '__main__':
    main()