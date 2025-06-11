import os
import json
from typing import List

import cv2
import torch
import dgl
from torchvision import transforms as T
from torch.utils.data import Dataset
from ultralytics import YOLO


class StreamDataset(Dataset):
    """Dataset that streams frames from RTSP cameras and performs on-the-fly
    person detection.
    """

    def __init__(self, cfg, feature_extractor, rtsp_urls: List[str] = None, dataset_dir: str = None, model_path: str = 'yolov8n.pt'):
        self.cfg = cfg
        self.device = self.cfg.MODEL.DEVICE
        self.feature_extractor = feature_extractor

        if rtsp_urls is None:
            rtsp_urls = [
                'rtsp://10.1.1.227:7070/stream1',
                'rtsp://10.1.1.227:7070/stream2',
                'rtsp://10.1.1.227:7070/stream3',
                'rtsp://10.1.1.227:7070/stream4'
            ]
        self.caps = [cv2.VideoCapture(u) for u in rtsp_urls]
        self.n_cams = len(self.caps)

        if dataset_dir is None:
            dataset_dir = os.path.join(cfg.DATASET.DIR, cfg.DATASET.NAME, cfg.DATASET.SEQUENCE[0])
        with open(os.path.join(dataset_dir, 'metainfo.json')) as fp:
            meta = json.load(fp)
        seq_name = cfg.DATASET.SEQUENCE[0]
        self.homography = torch.tensor(meta[seq_name]['homography'], dtype=torch.float32)

        self.detector = YOLO(model_path)
        self.frame_id = 0
        self.total_frames = cfg.DATASET.TOTAL_FRAMES if cfg.DATASET.TOTAL_FRAMES > 0 else 10 ** 9

        self.transform = T.Resize(self.cfg.FE.INPUT_SIZE)

    def __len__(self):
        return self.total_frames

    def _read_frames(self):
        imgs = []
        for cap in self.caps:
            ret, img = cap.read()
            if not ret:
                return None
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            imgs.append(img)
        return imgs

    def _detect(self, img):
        results = self.detector(img, verbose=False)[0]
        dets = []
        for box in results.boxes.data.cpu().numpy():
            x1, y1, x2, y2, score, cls = box
            if int(cls) != 0:
                continue
            dets.append([int(x1), int(y1), int(x2 - x1), int(y2 - y1)])
        return dets

    def __getitem__(self, index):
        imgs = self._read_frames()
        if imgs is None:
            raise IndexError

        detections = []
        for cid, img in enumerate(imgs):
            dets = self._detect(img)
            for d in dets:
                detections.append((*d, cid))

        n_node = len(detections)
        if n_node == 0:
            return dgl.graph(([], []), idtype=torch.int32, device=self.device), None, None

        H, W = self.cfg.FE.INPUT_SIZE
        cIDs = torch.zeros(n_node, 1, dtype=torch.int8)
        fIDs = torch.full((n_node, 1), self.frame_id, dtype=torch.int16)
        tIDs = torch.full((n_node, 1), -1, dtype=torch.int16)
        tIDs_pred = torch.full((n_node, 1), -1, dtype=torch.int16)
        bboxs = torch.zeros(n_node, 4, dtype=torch.float32)
        projs = torch.zeros(n_node, 3, dtype=torch.float32)
        bdets = torch.zeros(n_node, 3, H, W, dtype=torch.float32)

        idx = 0
        for cid, img in enumerate(imgs):
            dets = [d for d in detections if d[4] == cid]
            for x, y, w, h, _ in dets:
                proj = None
                if self.cfg.DATASET.NAME in ['Wildtrack']:
                    proj = torch.matmul(torch.linalg.inv(self.homography[cid]), torch.tensor([x + w / 2, y + h, 1.], dtype=torch.float32))
                else:
                    proj = torch.matmul(self.homography[cid], torch.tensor([x + w / 2, y + h, 1.], dtype=torch.float32))
                projs[idx] = proj / proj[-1]

                crop = img[y:y + h, x:x + w]
                crop = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
                crop = T.ToTensor()(crop)
                crop = self.transform(crop)
                bdets[idx] = crop

                cIDs[idx] = cid
                bboxs[idx] = torch.tensor([x, y, w, h])
                idx += 1

        features = self.feature_extractor(bdets)

        nodes_attr = {
            'cID': cIDs.to(self.device),
            'fID': fIDs.to(self.device),
            'tID': tIDs.to(self.device),
            'tID_pred': tIDs_pred.to(self.device),
            'bbox': bboxs.to(self.device),
            'feat': features.to(self.device),
            'proj': projs.to(self.device)
        }

        g = dgl.graph(([], []), idtype=torch.int32, device=self.device)
        g.add_nodes(n_node, nodes_attr)

        u, v = [], []
        for n in range(n_node):
            u += [n] * n_node
            v += list(range(n_node))
        g.add_edges(u, v)

        g_cID = g.ndata['cID']
        _from = g.edges()[0].type(torch.long)
        _to = g.edges()[1].type(torch.long)
        li = torch.where(g_cID[_from] == g_cID[_to])[0]
        if len(li) > 0:
            g.remove_edges(list(li))

        node_feature = g.ndata['feat']
        projs = g.ndata['proj']
        u = g.edges()[0].type(torch.long)
        v = g.edges()[1].type(torch.long)
        edge_feature = torch.vstack((
            torch.pairwise_distance(node_feature[u], node_feature[v]),
            1 - torch.cosine_similarity(node_feature[u], node_feature[v]),
            torch.pairwise_distance(projs[u, :2], projs[v, :2], p=1),
            torch.pairwise_distance(projs[u, :2], projs[v, :2], p=2)
        )).T
        g.edata['embed'] = edge_feature

        self.frame_id += 1
        return g, node_feature, edge_feature
