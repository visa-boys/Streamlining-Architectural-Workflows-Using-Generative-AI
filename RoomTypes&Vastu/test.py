from collections import defaultdict, Counter
from torch_geometric.data import Data
from torch_geometric.data import Dataset
import torch
import numpy as np
import pickle


class FloorplanGraphDataset(Dataset):
    def __init__(self, path, split=None):
        super(FloorplanGraphDataset, self).__init__()
        self.path = path
        with open(self.path, 'rb') as f:
            loaded_data = pickle.load(f)
        self.subgraphs = loaded_data
        self.subgraphs = self.filter_graphs(self.subgraphs)
        if split=='train':
            self.subgraphs = self.subgraphs[:120000]
        elif split=='test':
            self.subgraphs = self.subgraphs[120000:]    
        num_nodes = defaultdict(int)
        for g in self.subgraphs:
            labels = g[0] 
            if len(labels) > 0:
                num_nodes[len(labels)] += 1
        print(f'Number of graphs: {len(self.subgraphs)}')
        print(f'Number of graphs by rooms: {num_nodes}')
        
    def len(self):
        return len(self.subgraphs)

    def get(self, index, bbs=False):
        graph = self.subgraphs[index]
        labels = np.array(graph[0])
        rooms_bbs = np.array(graph[1])
        temp=rooms_bbs
#         edge2node = [item for sublist in graph[3] for item in sublist]
#         node_doors = np.array(edge2node)[graph[4]]
#         doors_count = Counter(node_doors)
        features = []
        rooms_bbs_new = []
        for i, bb in enumerate(rooms_bbs):
            x0, y0 = bb[0], bb[1]
            x1, y1 = bb[2], bb[3]
            #temp.append([x0,y0,x1,y1])
            xmin, ymin = min(x0, x1), min(y0, y1)
            xmax, ymax = max(x0, x1), max(y0, y1)
            l, b = xmax - xmin, ymax - ymin
            area = l*b
            if l<b:
                l, b = b, l
            features.append([area, l, b, 0, 0]) 
            rooms_bbs_new.append(np.array([xmin, ymin, xmax, ymax]))
        rooms_bbs = np.stack(rooms_bbs_new)
        intersect = self.intersect(rooms_bbs,rooms_bbs)
        for i in range(len(rooms_bbs)):
            for j in range(i+1,len(rooms_bbs)):
                if intersect[i,j]>0.7*intersect[j,j]:
                    if intersect[i,i]>intersect[j,j]: #is i a parent
                        features[i][4] = 1
                        features[j][3] = 1
                    else:   # i is child
                        features[i][3] = 1
                        features[j][4] = 1
                if intersect[i,j]>0.7*intersect[i,i]:
                    if intersect[j,j]>intersect[i,i]: 
                        features[j][4] = 1
                        features[i][3] = 1
                    else:
                        features[j][3] = 1
                        features[i][4] = 1

        rooms_bbs = rooms_bbs/256.0

        tl = np.min(rooms_bbs[:, :2], 0)
        br = np.max(rooms_bbs[:, 2:], 0)
        shift = (tl+br)/2.0 - 0.5
        rooms_bbs[:, :2] -= shift
        rooms_bbs[:, 2:] -= shift
        tl -= shift
        br -= shift
        edges = self.build_graph(rooms_bbs,temp) 
#         labels = labels - 1
#         labels[labels>=5] = labels[labels>=5] - 1
        x = torch.tensor(features, dtype=torch.float)
        edge_index = torch.tensor(edges.T, dtype=torch.long)
        y = torch.tensor(labels, dtype=torch.long)
        d = Data(x=x, edge_index=edge_index, y=y)
        if bbs:
            return d, rooms_bbs
        return d

    def build_graph(self, bbs,temp):
        edges = []
        for k in range(len(bbs)):
            for l in range(len(bbs)):
                if l > k:
                    bb0 = bbs[k]
                    bb1 = bbs[l]
                    #print(bbs,temp)
                    bb2 = temp[k]
                    bb3 = temp[l]
                    if self.is_adjacent(bb0, bb1) and  self.manhattam(bb2,bb3):
                        edges.append([k, l])
                        edges.append([l, k])
        edges = np.array(edges)
        return edges

    def filter_graphs(self, graphs):
        new_graphs = []
        for g in graphs:       
            labels = g[0]
            rooms_bbs = g[1]
            # discard broken samples
            check_none = np.sum([bb is None for bb in rooms_bbs])
            #check_node = np.sum([nd == 0 for nd in labels])
            if (len(labels) < 2) or (check_none > 0):
                continue
            new_graphs.append(g)
        return new_graphs

    def is_adjacent(self, box_a, box_b, threshold=0.03):
        
        x0, y0, x1, y1 = box_a
        x2, y2, x3, y3 = box_b

        h1, h2 = x1-x0, x3-x2
        w1, w2 = y1-y0, y3-y2

        xc1, xc2 = (x0+x1)/2.0, (x2+x3)/2.0
        yc1, yc2 = (y0+y1)/2.0, (y2+y3)/2.0

        delta_x = np.abs(xc2-xc1) - (h1 + h2)/2.0
        delta_y = np.abs(yc2-yc1) - (w1 + w2)/2.0

        delta = max(delta_x, delta_y)

        return delta < threshold

    def intersect(self, A,B):
        A, B = A[:,None], B[None]
        low = np.s_[...,:2]
        high = np.s_[...,2:]
        A,B = A.copy(),B.copy()
        A[high] += 1; B[high] += 1
        intrs = (np.maximum(0,np.minimum(A[high],B[high])
                            -np.maximum(A[low],B[low]))).prod(-1)
        return intrs #/ ((A[high]-A[low]).prod(-1)+(B[high]-B[low]).prod(-1)-intrs)
    def manhattam(self,box_a, box_b):
        x0, y0, x1, y1 = box_a
        x2, y2, x3, y3 = box_b

        if x2>=x1 and y2<=y1:
            return False
        elif x2>=x1 and y3>=y0:
            return False
        elif x3<=x0 and y3>=y0:
            return False
        elif x3<=x0 and y2<=y1:
            return False
        else:
            return True
MOD_ROOM_CLASS = {0: "Living room", 
                1: "Master oom",
                2: "Kitchen",
                3: "Bathroom",
                4: "Dining room",
                5: "Child room",
                6: "Study room",
                7: "Second room",
                8: "Guest room",
                9: "Balcony",
                10: "Entrance",
                11: "Storage",
                12: "Wall-in"}
def visualize(d, bbs=None):
    G = to_networkx(d, to_undirected=True)
    plt.figure(figsize=(7,7))
    plt.axis('off')
    labels = {i: MOD_ROOM_CLASS[int(d.y[i])] for i in range(len(d.y))}
    c = plt.get_cmap('tab20').colors
    color = [c[i] for i in d.y]
    nx.draw_networkx(G, pos=nx.spring_layout(G, seed=42), with_labels=True, labels=labels, node_color=color, cmap='Dark2')
    plt.show()
    if bbs is not None:
        plt.figure(figsize=(7,7))
        for i, (xmin, ymin, xmax, ymax) in enumerate(bbs):
            rect = patches.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, edgecolor='k', facecolor=c[d.y[i]], alpha=0.9)
            plt.gca().add_patch(rect)
        plt.show()
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
from torch_geometric.utils import to_networkx
import networkx as nx
#from dataset import MOD_ROOM_CLASS

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
def accuracy(model, dataloader):
    correct = 0
    num_nodes = 0
    model.to(device)
    model.eval()
    for data in dataloader:
        data = data.to(device)
        out = model(data.x, data.edge_index)
        pred = out.argmax(1)
        correct += sum(pred==data.y)
        num_nodes += data.num_nodes
    return (correct/num_nodes).item()
import torch
from torch.nn import Linear
import torch.nn.functional as F

class Model(torch.nn.Module):

    def __init__(self, layer_type, n_hidden=2):
        super(Model, self).__init__()
        torch.manual_seed(42)
        self.is_mlp = True if layer_type.__name__=='Linear' else False
        self.layer1 = layer_type(5, 16)
        self.layer2 = torch.nn.ModuleList()
        for _ in range(n_hidden-1):
            self.layer2.append(layer_type(16,16))
        self.classifier = Linear(16, 13)

    def forward(self, x, edge_index):
        h = self.layer1(x) if self.is_mlp else self.layer1(x, edge_index)
        h = F.relu(h)
        for layer in self.layer2:
            h = layer(h) if self.is_mlp else layer(h, edge_index)
            h = F.relu(h)
        out = self.classifier(h)
        return out
import torch
import numpy as np
#from dataset import FloorplanGraphDataset
#from utils import  accuracy
#from model import Model
from torch.nn import Linear
from torch_geometric.data import DataLoader
from torch_geometric.nn import SAGEConv, GATConv, GCNConv, TAGConv
import pathlib
#import argparse

model_type = 'gcn'  # 'mlp', 'gcn', 'gat', 'sage', 'tagcn'
hidden_layers = 2
num_epochs = 100
learning_rate = 0.004
step_size = 10
gamma = 0.8
batch_size = 128
outpath = '/kaggle/working/results'
dataset_file = '/kaggle/input/house-clean-data-rplan/clean_data.pkl'
models = {
    'mlp': Linear,
    'gcn': GCNConv,
    'gat': GATConv,
    'sage': SAGEConv,
    'tagcn': TAGConv,
}
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(device)

outpath = pathlib.Path(outpath)
outpath.mkdir(parents=True, exist_ok=True)

torch.manual_seed(42)
model = Model(layer_type=models[model_type], n_hidden=hidden_layers)
print(model)
model = model.to(device)
dataset = FloorplanGraphDataset(path=dataset_file, split=None)
# for i in range(65000):
#     print(i)
#     dataset[i].to(device)
train = [dataset[i].to(device) for i in range(65000)]
trainloader = DataLoader(train, batch_size=batch_size, shuffle=True)
trainloader2 = DataLoader(train, batch_size=65000)
test = [dataset[i].to(device) for i in range(65000,80788)]
testloader = DataLoader(test, batch_size=15788)
num_epochs = num_epochs
lr = learning_rate
step_size = step_size
gamma = gamma
outpath = '/kaggle/working/'

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

loss_ep = []
te_acc_ep = []
tr_acc_ep = []
for epoch in range(num_epochs):
    model.train()
    loss = 0
    for data in trainloader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss_ = criterion(out, data.y)
        loss_.backward()
        optimizer.step()
        loss += loss_.item()
    exp_lr_scheduler.step()
    loss/=len(train)
    tr_acc = accuracy(model, trainloader2)
    te_acc = accuracy(model, testloader)
    loss_ep.append(loss)
    tr_acc_ep.append(tr_acc)
    te_acc_ep.append(te_acc)
    print(f'Epoch [{epoch+1}/{num_epochs}] Loss: {loss:.10f}, Train Acc: {tr_acc:.6f}, Test Acc: {te_acc:.6f}')
