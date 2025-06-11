import argparse
import os

from configs import cfg
from src.tracker import Tracker
from src.datasets.stream_dataset import StreamGraphDataset
from src.utils.tools import udf_collate_fn, make_dir
from torch.utils.data import DataLoader


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", default="configs/MeetingRoom.yml", type=str)
    parser.add_argument("--urls", nargs='+', required=True, help="RTSP stream urls")
    parser.add_argument("--frames", type=int, default=0, help="number of frames to process, 0 for infinity")
    parser.add_argument("opts", help="Modify config options using the command-line", default=None, nargs=argparse.REMAINDER)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.config_file:
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID

    tracker = Tracker(cfg)
    dataset = StreamGraphDataset(cfg, args.urls, tracker.feature_extractor, max_frames=args.frames)
    dataloader = DataLoader(dataset, 1, collate_fn=udf_collate_fn)

    ckpt = tracker.load_param("test")

    visualize_dir = None
    if cfg.OUTPUT.VISUALIZE:
        visualize_dir = os.path.join(tracker.output_dir, 'visualize')
        make_dir(visualize_dir)

    tracker._test_one_epoch(dataloader, ckpt['L'], visualize_dir)


if __name__ == "__main__":
    main()
