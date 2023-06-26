from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import codecs
sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())

import pandas as pd
import json
import misc.utils as utils
from pycocotools.coco import COCO
from json import encoder
encoder.FLOAT_REPR = lambda o: format(o, '.3f')


if __name__ == '__main__':
    noc_object = ['bus', 'bottle', 'couch', 'microwave', 'pizza', 'racket', 'suitcase', 'zebra']
    dataset = ['data/noc_coco/captions_split_set_%s_val_test_novel2014.json'%item for item in noc_object]

    pred_file = 'save/preds.json'
    ann_file = 'data/annotations/captions_val2014.json'
    # prediction
    with open(pred_file, 'rb') as f:
        pred = json.load(f)

    score_dict = {}
    coco = COCO(ann_file)
    gt = coco.dataset['annotations']

    for idx in range(len(dataset)):
        lang_stats = utils.language_eval(dataset[idx], pred, noc_object[idx], 'test', '')
        score_dict[str(noc_object[idx])] = lang_stats
    
    df = pd.DataFrame()
    for k, v in score_dict.items():
        df[k] = v.values()
    df.index = v.keys()

    df.to_csv('save/res.csv')
