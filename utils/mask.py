###################################################################
# File Name: mask.py
# Author: Zhongdao Wang
# mail: wcd17@mails.tsinghua.edu.cn
# Created Time: Tue Feb  9 10:05:47 2021
###################################################################

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import pdb
import cv2
import torch
import numpy as np
# from sklearn.metrics import jaccard_similarity_score
from sklearn.metrics import jaccard_score as jaccard_similarity_score
import pycocotools.mask as mask_utils

import matplotlib.pyplot as plt

def coords2bbox(coords, extend=2):
    """
    INPUTS:
     - coords: coordinates of pixels in the next frame
    """
    center = torch.mean(coords, dim=0) # b * 2
    center = center.view(1,2)
    center_repeat = center.repeat(coords.size(0),1)

    dis_x = torch.sqrt(torch.pow(coords[:,0] - center_repeat[:,0], 2))
    dis_x = max(torch.mean(dis_x, dim=0).detach(),1)
    dis_y = torch.sqrt(torch.pow(coords[:,1] - center_repeat[:,1], 2))
    dis_y = max(torch.mean(dis_y, dim=0).detach(),1)

    left = center[:,0] - dis_x*extend
    right = center[:,0] + dis_x*extend
    top = center[:,1] - dis_y*extend
    bottom = center[:,1] + dis_y*extend

    return (top.item(), left.item(), bottom.item(), right.item())

def mask2box(masks):
    boxes = []
    for mask in masks:
        m = mask[0].nonzero().float()
        if m.numel() > 0:
            box = coords2bbox(m, extend=2)
        else:
            box = (-1,-1,10,10)
        boxes.append(box)
    return np.asarray(boxes)

    
def temp_interp_mask(maskseq, T):
    '''
    maskseq: list of elements (RLE_mask, timestamp)
    return list of RLE_mask, length of list is T
    '''
    size = maskseq[0][0]['size']
    blank_mask = np.asfortranarray(np.zeros(size).astype(np.uint8))
    blank_mask = mask_utils.encode(blank_mask)
    blank_mask['counts'] = blank_mask['counts'].decode('ascii')
    ret = [blank_mask,] * T
    for m, t in maskseq:
        ret[t] = m
    return ret

def mask_seq_jac(sa, sb):
    j = np.zeros((len(sa), len(sb)))
    for ia, a in enumerate(sa):
        for ib, b in enumerate(sb):
            ious = [mask_utils.iou([at], [bt], [False,]) for (at, bt) in zip(a,b)]
            tiou = np.mean(ious)
            j[ia, ib] = tiou
    return j
        

def skltn2mask(skltn, size):
    h, w = size
    mask = np.zeros((h,w))
    
    dskltn = dict()
    for s in skltn:
        dskltn[s['id'][0]] = (int(s['x'][0]), int(s['y'][0]))
    if len(dskltn)==0:
        return mask
    trunk_polygon = list()
    for k in np.array([3,4,10,13,9])-1:
        p = dskltn.get(k, None)
        if not p is None:
            trunk_polygon.append(p)
    trunk_polygon = np.asarray(trunk_polygon, 'int32')
    if len(trunk_polygon) > 2:
        cv2.fillConvexPoly(mask, trunk_polygon, 1)

    xmin = np.min([dskltn[k][0] for k in dskltn])
    xmax = np.max([dskltn[k][0] for k in dskltn])
    ymin = np.min([dskltn[k][1] for k in dskltn])
    ymax = np.max([dskltn[k][1] for k in dskltn])
    line_width = np.max([int(np.max([xmax-xmin, ymax-ymin, 0])/20),8])


    skeleton = [[10, 11], [11, 12], [9,8], 
                [8,7], [10, 13], [9, 13], 
                [13, 15], [10,4], [4,5], 
                [5,6], [9,3], [3,2], [2,1]]
    

    for sk in skeleton:
        st = dskltn.get(sk[0]-1, None)
        ed = dskltn.get(sk[1]-1, None)
        if st is None or ed is None:
            continue
        cv2.line(mask, st, ed, color=1, thickness=line_width)
    
    #dmask = cv2.resize(mask, (w//8, h//8), interpolation=cv2.INTER_NEAREST)
    #pdb.set_trace()
    
    return mask

def pts2array(pts):
    arr = np.zeros((15,3))
    for s in pts:
        arr[s['id'][0]][0] = int(s['x'][0])
        arr[s['id'][0]][1] = int(s['y'][0])
        arr[s['id'][0]][2] = s['score'][0]
    return arr

    
