import cv2
import torch
import dgl
from torch.utils.data import Dataset
from torchvision import transforms as T
from ultralytics import YOLO


class StreamGraphDataset(Dataset):
    """Dataset that reads frames from RTSP streams and builds graphs for tracking."""

    def __init__(self, cfg, urls, feature_extractor, max_frames=0):
        self.cfg = cfg
        self.device = self.cfg.MODEL.DEVICE
        self.caps = [cv2.VideoCapture(u) for u in urls]
        self.feature_extractor = feature_extractor
        self.detector = YOLO(self.cfg.MODEL.DETECTION)
        self.max_frames = max_frames if max_frames > 0 else float('inf')
        self.frame_id = 0

    def __len__(self):
        return int(self.max_frames)

    def __getitem__(self, index):
        imgs = []
        for cap in self.caps:
            ret, img = cap.read()
            if not ret:
                raise RuntimeError('Failed to read frame from stream')
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            imgs.append(img)

        det_imgs = []
        nodes = []
        (H, W) = self.cfg.FE.INPUT_SIZE
        for cid, img in enumerate(imgs):
            results = self.detector(img)
            boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
            for x1, y1, x2, y2 in boxes:
                det = img[y1:y2, x1:x2]
                det = cv2.resize(det, (W, H))
                det_imgs.append(T.ToTensor()(det))
                nodes.append([cid, x1, y1, x2 - x1, y2 - y1])

        if len(det_imgs) == 0:
            graph = dgl.graph(([], []), idtype=torch.int32, device=self.device)
            self.frame_id += 1
            return graph, torch.empty(0), torch.empty(0)

        bdets = torch.stack(det_imgs)
        feat = self.feature_extractor(bdets).to(self.device)

        num_nodes = len(nodes)
        graph = dgl.graph(([], []), idtype=torch.int32, device=self.device)
        projs = torch.zeros(num_nodes, 3, dtype=torch.float32)
        cIDs = torch.tensor([n[0] for n in nodes], dtype=torch.int16).unsqueeze(1)
        bbox = torch.tensor([n[1:] for n in nodes], dtype=torch.float32)
        fIDs = torch.full((num_nodes, 1), self.frame_id, dtype=torch.int16)
        tIDs = torch.full((num_nodes, 1), -1, dtype=torch.int16)
        tIDs_pred = torch.full((num_nodes, 1), -1, dtype=torch.int16)
        graph.add_nodes(num_nodes, {
            'cID': cIDs.to(self.device),
            'fID': fIDs.to(self.device),
            'tID': tIDs.to(self.device),
            'tID_pred': tIDs_pred.to(self.device),
            'bbox': bbox.to(self.device),
            'feat': feat,
            'proj': projs.to(self.device)
        })

        u, v = [], []
        for n in range(num_nodes):
            u += [n] * num_nodes
            v += list(range(num_nodes))
        graph.add_edges(u, v)

        _from = graph.edges()[0].type(torch.long)
        _to = graph.edges()[1].type(torch.long)
        li = torch.where(cIDs[_from] == cIDs[_to])[0]
        if len(li) > 0:
            graph.remove_edges(list(li))

        node_feature = graph.ndata['feat']
        projs = graph.ndata['proj']
        edge_feature = torch.vstack((
            torch.pairwise_distance(node_feature[_from], node_feature[_to]),
            1 - torch.cosine_similarity(node_feature[_from], node_feature[_to]),
            torch.pairwise_distance(projs[_from, :2], projs[_to, :2], p=1),
            torch.pairwise_distance(projs[_from, :2], projs[_to, :2], p=2),
        )).T
        graph.edata['embed'] = edge_feature

        self.frame_id += 1
        return graph, node_feature, edge_feature
