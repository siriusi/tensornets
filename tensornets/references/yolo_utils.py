from __future__ import absolute_import
from __future__ import division

import os
import numpy as np
import tensorflow as tf

try:
    from .darkflow_utils.get_boxes import yolov3_box
    from .darkflow_utils.get_boxes import yolov2_box
except ImportError:
    yolov3_box = None
    yolov2_box = None

try:
    xrange          # Python 2
except NameError:
    xrange = range  # Python 3


with open(os.path.join(os.path.dirname(__file__), 'coco.names'), 'r') as f:
    labels_coco = [line.rstrip() for line in f.readlines()]

with open(os.path.join(os.path.dirname(__file__), 'voc.names'), 'r') as f:
    labels_voc = [line.rstrip() for line in f.readlines()]

bases = dict()
bases['yolov3'] = {'anchors': [10., 13., 16., 30., 33., 23., 30., 61.,
                               62., 45., 59., 119., 116., 90., 156., 198.,
                               373., 326.]}
bases['yolov3coco'] = bases['yolov3']
bases['yolov3voc'] = bases['yolov3']
bases['yolov2'] = {'anchors': [0.57273, 0.677385, 1.87446, 2.06253, 3.33843,
                               5.47434, 7.88282, 3.52778, 9.77052, 9.16828]}
bases['yolov2voc'] = {'anchors': [1.3221, 1.73145, 3.19275, 4.00944, 5.05587,
                                  8.09892, 9.47112, 4.84053, 11.2364, 10.0071]}
bases['tinyyolov2voc'] = {'anchors': [1.08, 1.19, 3.42, 4.41, 6.63,
                                      11.38, 9.42, 5.11, 16.62, 10.52]}


def opts(model_name):
    opt = bases[model_name].copy()
    opt.update({'num': len(opt['anchors']) // 2})
    if 'voc' in model_name:
        opt.update({'classes': len(labels_voc), 'labels': labels_voc})
    else:
        opt.update({'classes': len(labels_coco), 'labels': labels_coco})
    return opt


def parse_box(b, t, w, h):
    idx = np.argmax(b.probs)
    score = b.probs[idx]
    if score > t:
        x1 = int((b.x - b.w / 2) * w)
        y1 = int((b.y - b.h / 2) * h)
        x2 = int((b.x + b.w / 2) * w)
        y2 = int((b.y + b.h / 2) * h)
        if x1 < 0:
            x1 = 0
        if x2 > w - 1:
            x2 = w - 1
        if y1 < 0:
            y1 = 0
        if y2 > h - 1:
            y2 = h - 1
        return idx, (x1, y1, x2, y2, score)
    else:
        return None, None


def get_v3_boxes(opts, outs, source_size, threshold=0.1):
    h, w = source_size
    boxes = [[] for _ in xrange(opts['classes'])]
    opts['thresh'] = threshold
    opts['in_size'] = (416, 416)
    for i in range(3):
        opts['out_size'] = list(outs[i][0].shape)
        opts['anchor_idx'] = 6 - 3 * i
        results = yolov3_box(opts, outs[i][0].copy())
        for b in results:
            idx, box = parse_box(b, threshold, w, h)
            if idx is not None:
                boxes[idx].append(box)
    for i in xrange(opts['classes']):
        boxes[i] = np.asarray(boxes[i], dtype=np.float32)
    return boxes


def get_v2_boxes(opts, outs, source_size, threshold=0.1):
    h, w = source_size
    boxes = [[] for _ in xrange(opts['classes'])]
    opts['thresh'] = threshold
    opts['out_size'] = list(outs[0].shape)
    results = yolov2_box(opts, outs[0].copy())
    for b in results:
        idx, box = parse_box(b, threshold, w, h)
        if idx is not None:
            boxes[idx].append(box)
    for i in xrange(opts['classes']):
        boxes[i] = np.asarray(boxes[i], dtype=np.float32)
    return boxes


def v2_placeholders(opts, out_shape):
    height, width = out_shape
    size1 = [None, height * width, opts['num'], opts['classes']]
    size2 = [None, height * width, opts['num']]
    return {
        # return the below placeholders
        'probs': tf.placeholder(tf.float32, size1),
        'confs': tf.placeholder(tf.float32, size2),
        'coord': tf.placeholder(tf.float32, size2 + [4]),
        # weights term for L2 loss
        'proid': tf.placeholder(tf.float32, size1),
        # material calculating IOU
        'areas': tf.placeholder(tf.float32, size2),
        'upleft': tf.placeholder(tf.float32, size2 + [2]),
        'botright': tf.placeholder(tf.float32, size2 + [2])
    }


def expit_tensor(x):
    return 1. / (1. + tf.exp(-x))


def v2_loss(opts, outs):
    sprob = 1.
    sconf = 5.
    snoob = 1.
    scoor = 1.
    H = outs.shape[1].value
    W = outs.shape[2].value
    B, C = opts['num'], opts['classes']
    HW = H * W # number of grid cells
    anchors = opts['anchors']

    # Extract the coordinate prediction from net.out
    net_out_reshape = tf.reshape(outs, [-1, H, W, B, (4 + 1 + C)])
    coords = net_out_reshape[:, :, :, :, :4]
    coords = tf.reshape(coords, [-1, H*W, B, 4])
    adjusted_coords_xy = expit_tensor(coords[:,:,:,0:2])
    adjusted_coords_wh = tf.sqrt(tf.exp(coords[:,:,:,2:4]) * np.reshape(anchors, [1, 1, B, 2]) / np.reshape([W, H], [1, 1, 1, 2]))
    coords = tf.concat([adjusted_coords_xy, adjusted_coords_wh], 3)

    adjusted_c = expit_tensor(net_out_reshape[:, :, :, :, 4])
    adjusted_c = tf.reshape(adjusted_c, [-1, H*W, B, 1])

    adjusted_prob = tf.nn.softmax(net_out_reshape[:, :, :, :, 5:])
    adjusted_prob = tf.reshape(adjusted_prob, [-1, H*W, B, C])

    adjusted_net_out = tf.concat([adjusted_coords_xy, adjusted_coords_wh, adjusted_c, adjusted_prob], 3)

    wh = tf.pow(coords[:,:,:,2:4], 2) * np.reshape([W, H], [1, 1, 1, 2])
    area_pred = wh[:,:,:,0] * wh[:,:,:,1]
    centers = coords[:,:,:,0:2]
    floor = centers - (wh * .5)
    ceil  = centers + (wh * .5)

    # calculate the intersection areas
    intersect_upleft   = tf.maximum(floor, outs.placeholders['upleft'])
    intersect_botright = tf.minimum(ceil , outs.placeholders['botright'])
    intersect_wh = intersect_botright - intersect_upleft
    intersect_wh = tf.maximum(intersect_wh, 0.0)
    intersect = tf.multiply(intersect_wh[:,:,:,0], intersect_wh[:,:,:,1])

    # calculate the best IOU, set 0.0 confidence for worse boxes
    iou = tf.truediv(intersect, outs.placeholders['areas'] + area_pred - intersect)
    best_box = tf.equal(iou, tf.reduce_max(iou, [2], True))
    best_box = tf.to_float(best_box)
    confs = tf.multiply(best_box, outs.placeholders['confs'])

    # take care of the weight terms
    conid = snoob * (1. - confs) + sconf * confs
    weight_coo = tf.concat(4 * [tf.expand_dims(confs, -1)], 3)
    cooid = scoor * weight_coo
    weight_pro = tf.concat(C * [tf.expand_dims(confs, -1)], 3)
    proid = sprob * weight_pro

    fetch = [outs.placeholders['probs'], confs, conid, cooid, proid]
    true = tf.concat([outs.placeholders['coord'], tf.expand_dims(confs, 3), outs.placeholders['probs'] ], 3)
    wght = tf.concat([cooid, tf.expand_dims(conid, 3), proid ], 3)

    loss = tf.pow(adjusted_net_out - true, 2)
    loss = tf.multiply(loss, wght)
    loss = tf.reshape(loss, [-1, H*W*B*(4 + 1 + C)])
    loss = tf.reduce_sum(loss, 1)
    return .5 * tf.reduce_mean(loss)
