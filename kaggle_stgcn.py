"""
Stage 2+3: naive tracker -> ST-GCN++ classification.

Continues from bu_pose_extractor.py's .pkl output. Skips BUCTD/ByteTrack for
now and borrows pyskl's own naive greedy tracker (demo/demo_skeleton.py)
verbatim, then feeds the result straight into a downloaded ST-GCN++ hrnet
checkpoint

Run from inside pyskl repo clone (needs its configs/ + label_map/
files and the `pyskl` package importable), e.g.:

    cd /kaggle/working/pyskl
    python /path/to/track_and_classify.py /kaggle/working/bu_poses.pkl \\
        --video /kaggle/working/snip_videoplayback.mp4 --device cpu

--------------------------------------------------------------------------
IMPORTANT LIMITATION, carried over from the ST-GCN++ input-format check:
FormatGCNInput caps this at <= 2 people PER RUN (num_person read straight
from the config, defaulting to 2 like pyskl's own script does). pose_tracking
here will keep only the 2 tracks
and silently drop everyone else, this
gives a working result on 2 people. Grouping
the rest into additional per-person / per-pair runs is follow-up work
--------------------------------------------------------------------------
"""

import argparse
import os.path as osp
import pickle

import cv2
import mmcv
import numpy as np
from scipy.optimize import linear_sum_assignment

# pyskl must be importable (run from the pyskl repo root, or `pip install -e .` it).
from pyskl.apis import init_recognizer, inference_recognizer

DEFAULT_CONFIG = 'configs/stgcn++/stgcn++_ntu120_xsub_hrnet/j.py'
DEFAULT_CHECKPOINT = (
    'http://download.openmmlab.com/mmaction/pyskl/ckpt/stgcnpp/'
    'stgcnpp_ntu120_xsub_hrnet/j.pth'
)
DEFAULT_LABEL_MAP = 'tools/data/label_map/nturgbd_120.txt'


# ---------------------------------------------------------------------------
# Borrowed verbatim from pyskl/demo/demo_skeleton.py (dist_ske + pose_tracking).
# Operates purely on raw (17, 3) x,y,score arrays, doesn't care whether
# they came from inference_top_down_pose_model or inference_bottom_up_pose_model,
# since neither 'bbox' nor 'area' nor the instance-level 'score' is used here.
# ---------------------------------------------------------------------------
def dist_ske(ske1, ske2):
    dist = np.linalg.norm(ske1[:, :2] - ske2[:, :2], axis=1) * 2
    diff = np.abs(ske1[:, 2] - ske2[:, 2])
    return np.sum(np.maximum(dist, diff))


def pose_tracking(pose_results, max_tracks=2, thre=30):
    tracks, num_tracks = [], 0
    num_joints = None
    for idx, poses in enumerate(pose_results):
        if len(poses) == 0:
            continue
        if num_joints is None:
            num_joints = poses[0].shape[0]
        track_proposals = [t for t in tracks if t['data'][-1][0] > idx - thre]
        n, m = len(track_proposals), len(poses)
        scores = np.zeros((n, m))
        for i in range(n):
            for j in range(m):
                scores[i][j] = dist_ske(track_proposals[i]['data'][-1][1], poses[j])
        row, col = linear_sum_assignment(scores)
        for r, c in zip(row, col):
            track_proposals[r]['data'].append((idx, poses[c]))
        if m > n:
            for j in range(m):
                if j not in col:
                    num_tracks += 1
                    new_track = dict(data=[])
                    new_track['track_id'] = num_tracks
                    new_track['data'] = [(idx, poses[j])]
                    tracks.append(new_track)
    if num_joints is None:
        return None, None
    tracks.sort(key=lambda x: -len(x['data']))
    result = np.zeros((max_tracks, len(pose_results), num_joints, 3), dtype=np.float16)
    for i, track in enumerate(tracks[:max_tracks]):
        for item in track['data']:
            idx, pose = item
            result[i, idx] = pose
    return result[..., :2], result[..., 2]


def get_resized_shape(video_path, short_side=512):
    """Only need the frame size here, read frame 0."""
    vid = cv2.VideoCapture(video_path)
    ok, frame = vid.read()
    vid.release()
    if not ok:
        raise IOError(f'Could not read a frame from {video_path}')
    h, w = frame.shape[:2]
    new_w, new_h = mmcv.rescale_size((w, h), (short_side, np.inf))
    return new_h, new_w


def build_fake_anno(per_frame_results, img_shape):
    """Matches pyskl demo_skeleton.py's fake_anno schema: frame_dir/label/original_shape/start_index/modality
    are fixed placeholders pyskl itself uses for inference-only calls."""
    n_frames = len(per_frame_results)

    # bu_pose_extractor.py's dict schema -> the raw (17,3) arrays pose_tracking wants.
    tracking_inputs = [
        [np.concatenate([p['keypoints'], p['kpt_scores'][:, None]], axis=-1)
         for p in frame]
        for frame in per_frame_results
    ]

    fake_anno = dict(
        frame_dir='',
        label=-1,
        img_shape=img_shape,
        original_shape=img_shape,
        start_index=0,
        modality='Pose',
        total_frames=n_frames,
    )
    return fake_anno, tracking_inputs


def run(pkl_path, config=DEFAULT_CONFIG, checkpoint=DEFAULT_CHECKPOINT,
        label_map=DEFAULT_LABEL_MAP, device='cuda:0',
        video_path=None, short_side=512, img_shape=None):
    with open(pkl_path, 'rb') as f:
        per_frame_results = pickle.load(f)
    print(f'Loaded {len(per_frame_results)} frames from {pkl_path}')

    # img_shape resolution: needed for PreNormalize2D (see prior verification --
    # missing/wrong img_shape silently normalizes against a (1080, 1920) default).
    if img_shape is None:
        if video_path:
            img_shape = get_resized_shape(video_path, short_side=short_side)
            print(f'Resolved img_shape={img_shape} from {video_path} (short_side={short_side})')
        else:
            all_xy = np.concatenate(
                [p['keypoints'] for f in per_frame_results for p in f], axis=0)
            img_shape = (int(np.ceil(all_xy[:, 1].max())) + 1,
                        int(np.ceil(all_xy[:, 0].max())) + 1)
            print(f'WARNING: no --video/--img-shape given. Guessing img_shape='
                 f'{img_shape} from keypoint extent, this is a lower bound, '
                 f'not the true frame size, and will affect normalization '
                 f'accuracy. Pass --video or --img-shape H W if possible.')

    fake_anno, tracking_inputs = build_fake_anno(per_frame_results, img_shape)

    config_obj = mmcv.Config.fromfile(config)
    GCN_flag = 'GCN' in config_obj.model.type
    assert GCN_flag, f'{config} is not a GCN config (RecognizerGCN expected)'
    format_op = [op for op in config_obj.data.test.pipeline if op['type'] == 'FormatGCNInput'][0]
    GCN_nperson = format_op.get('num_person', 2)
    print(f'Config caps this run at max_tracks={GCN_nperson} people '
          f'(FormatGCNInput.num_person) -- see the module docstring for what '
          f'this means for a >{GCN_nperson}-person scene.')

    keypoint, keypoint_score = pose_tracking(tracking_inputs, max_tracks=GCN_nperson)
    if keypoint is None:
        print('No people detected in any frame -- nothing to classify.')
        return None
    fake_anno['keypoint'] = keypoint
    fake_anno['keypoint_score'] = keypoint_score

    print(f'Loading {config} + {checkpoint} on {device}...')
    model = init_recognizer(config, checkpoint, device=device)

    label_names = [x.strip() for x in open(label_map).readlines()]

    print('Running inference_recognizer...')
    top5 = inference_recognizer(model, fake_anno)

    print('\nTop-5 predictions:')
    for idx, score in top5:
        print(f'  {label_names[idx]:<40s} {score:.4f}')

    return top5


def parse_args():
    p = argparse.ArgumentParser(
        description='Naive tracker + ST-GCN++ classification on a bu_pose_extractor.py .pkl'
    )
    p.add_argument('pkl', help='path to the .pkl from bu_pose_extractor.py --out')
    p.add_argument('--config', default=DEFAULT_CONFIG,
                    help='pyskl GCN config path, relative to the pyskl repo root')
    p.add_argument('--checkpoint', default=DEFAULT_CHECKPOINT,
                    help='checkpoint path or URL (URLs are auto-downloaded/cached by pyskl)')
    p.add_argument('--label-map', default=DEFAULT_LABEL_MAP)
    p.add_argument('--device', default='cuda:0', help="'cuda:0' or 'cpu'")
    p.add_argument('--video', default=None,
                    help='original video, used only to resolve the true resized '
                         'frame size for img_shape (recommended)')
    p.add_argument('--short-side', type=int, default=512,
                    help='must match what bu_pose_extractor.py used')
    p.add_argument('--img-shape', type=int, nargs=2, default=None, metavar=('H', 'W'),
                    help='explicit override for img_shape, if you know it and skip --video')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    run(args.pkl, config=args.config, checkpoint=args.checkpoint,
        label_map=args.label_map, device=args.device, video_path=args.video,
        short_side=args.short_side,
        img_shape=tuple(args.img_shape) if args.img_shape else None)
