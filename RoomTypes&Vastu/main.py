import numpy as np
import torch
from torch_geometric.data import Data
def intersect(A,B):
    A, B = A[:,None], B[None]
    low = np.s_[...,:2]
    high = np.s_[...,2:]
    A,B = A.copy(),B.copy()
    A[high] += 1; B[high] += 1
    intrs = (np.maximum(0,np.minimum(A[high],B[high])
                        -np.maximum(A[low],B[low]))).prod(-1)
    return intrs #/ ((A[high]-A[low]).prod(-1)+(B[high]-B[low]).prod(-1)-intrs)

def build_graph(bbs,temp):
    edges = []
    for k in range(len(bbs)):
        for l in range(len(bbs)):
            if l > k:
                bb0 = bbs[k]
                bb1 = bbs[l]
                #print(bbs,temp)
                bb2 = temp[k]
                bb3 = temp[l]
                if is_adjacent(bb0, bb1) and  manhattam(bb2,bb3):
                    edges.append([k, l])
                    edges.append([l, k])
    edges = np.array(edges)
    return edges

def is_adjacent(box_a, box_b, threshold=0.03):
        
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

def manhattam(box_a, box_b):
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
def fn(rectangles,orientation):
    
    rooms_bbs = np.array(rectangles)
    temp=rooms_bbs
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
        features.append([area, l, b, 0, 0,orientation[i]]) 
        rooms_bbs_new.append(np.array([xmin, ymin, xmax, ymax]))
    rooms_bbs = np.stack(rooms_bbs_new)
    intersect_ = intersect(rooms_bbs,rooms_bbs)
    for i in range(len(rooms_bbs)):
        for j in range(i+1,len(rooms_bbs)):
            if intersect_[i,j]>0.7*intersect_[j,j]:
                if intersect_[i,i]>intersect_[j,j]: #is i a parent
                    features[i][4] = 1
                    features[j][3] = 1
                else:   # i is child
                    features[i][3] = 1
                    features[j][4] = 1
            if intersect_[i,j]>0.7*intersect_[i,i]:
                if intersect_[j,j]>intersect_[i,i]: 
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
    edges = build_graph(rooms_bbs,temp) 
    #         labels = labels - 1
    #         labels[labels>=5] = labels[labels>=5] - 1
    x = torch.tensor(features, dtype=torch.float)
    edge_index = torch.tensor(edges.T, dtype=torch.long)
    #y = torch.tensor(labels, dtype=torch.long)
    d = Data(x=x, edge_index=edge_index)
    return d