import json
import os
import re
import sys

# from python_utils import *
sys.path.append('tools/coco-caption/')
COCO_EVAL_PATH = '.tools/coco-caption/pycocotools'
sys.path.insert(0, COCO_EVAL_PATH)
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.eval import COCOEvalCap
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.spice.spice import Spice
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from pycocotools.coco import COCO

rm_word_dict = {'bus': ['bus', 'busses'],
                'bottle': ['bottle', 'bottles'],
                'couch': ['couch', 'couches', 'sofa', 'sofas'],
                'microwave': ['microwave', 'microwaves'],
                'pizza': ['pizza', 'pizzas'],
                'racket': ['racket', 'rackets', 'racquet', 'racquets'],
                'suitcase': ['luggage', 'luggages', 'suitcase', 'suitcases'],
                'zebra': ['zebra', 'zebras']}


def read_json(t_file):
    return json.load(open(t_file, 'r'))


class DCCScorer(COCOEvalCap):

    def get_dcc_scores(self):

        # imgIds = self.params['image_id']
        imgIds = self.cocoRes.getImgIds()
        gts = {}
        res = {}
        for imgId in imgIds:
            gts[imgId] = self.coco.imgToAnns[imgId]
            res[imgId] = self.cocoRes.imgToAnns[imgId]

        tokenizer = PTBTokenizer()
        gts = tokenizer.tokenize(gts)
        res = tokenizer.tokenize(res)
        scorers = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            (Meteor(), "METEOR"),
            (Rouge(), "ROUGE_L"),
            (Cider(), "CIDEr"),
            (Spice(), "SPICE")
        ]
        score_dict = {}
        for scorer, method in scorers:
            print('computing %s score...' % (scorer.method()))
            score, scores = scorer.compute_score(gts, res)
            if type(method) == list:
                for sc, scs, m in zip(score, scores, method):
                    score_dict[m] = sc
                    print("%s: %0.3f" % (m, sc))
            else:
                score_dict[method] = score
                print("%s: %0.3f" % (method, score))

        return score_dict


def split_sent(sent):
    sent = sent.lower()
    sent = re.sub('[^A-Za-z0-9\s]+', '', sent)
    return sent.split()


def F1(generated_json, novel_ids, train_ids, word):
    set_rm_words = set(rm_word_dict[word])
    gen_dict = {}
    for c in generated_json:
        gen_dict[c['image_id']] = c['caption']

    # true positive are sentences that contain match words and should
    tp = sum([1 for c in novel_ids if len(set_rm_words.intersection(set(split_sent(gen_dict[c])))) > 0])
    # false positive are sentences that contain match words and should not
    fp = sum([1 for c in train_ids if len(set_rm_words.intersection(set(split_sent(gen_dict[c])))) > 0])
    # false negative are sentences that do not contain match words and should
    fn = sum([1 for c in novel_ids if len(set_rm_words.intersection(set(split_sent(gen_dict[c])))) == 0])

    # precision = tp/(tp+fp)
    if tp > 0:
        precision = float(tp) / (tp + fp)
        # recall = tp/(tp+fn)
        recall = float(tp) / (tp + fn)
        # f1 = 2* (precision*recall)/(precision+recall)
        return 2 * (precision * recall) / (precision + recall)
    else:
        return 0.


def score_dcc(gt_template_novel, generation_result, words, dset, cache_path):
    score_dict_dcc = {}
    generated_sentences = generation_result
    f1_scores = 0

    all_gen = dict()
    for word in words:
        gt_json_novel = read_json(gt_template_novel % (word, dset))
        all_gen[word] = [c['image_id'] for c in gt_json_novel['annotations']]

    det_val_path = '%s/annotations/captions_val2014.json' % './data'

    coco = COCO(det_val_path)
    json.dump(generated_sentences, open(cache_path, 'w'))
    generation_coco = coco.loadRes(cache_path)
    dcc_evaluator = DCCScorer(coco, generation_coco)
    score_dict = dcc_evaluator.get_dcc_scores()
    os.remove(cache_path)

    out = {}
    for key in score_dict.keys():
        out[key] = score_dict[key]

    for word in words:
        gt_ids_novel = all_gen[word]
        gt_ids_train = [p for word2 in words if word2 != word for p in all_gen[word2]]

        f1_score = F1(generated_sentences, gt_ids_novel, gt_ids_train, word)
        print("F1 score for %s: %f" % (word, f1_score))
        f1_scores += f1_score

    print("########################################################################")
    out['F1'] = f1_scores / len(words)

    return out


def score_generation(gt_filename=None, generation_result=None):
    coco = COCO(gt_filename)
    generation_coco = coco.loadRes(generation_result)
    # coco_evaluator = COCOEvalCap(coco, generation_coco, 'noc_test_freq')
    coco_evaluator = COCOEvalCap(coco, generation_coco)
    coco_evaluator.evaluate()


def save_json_coco_format(caps, save_name):
    def get_coco_id(im_name):
        coco_id = int(im_name.split('/')[-1].split('_')[-1].split('.jpg')[0])
        return coco_id

    coco_format_caps = [{'caption': value, 'image_id': get_coco_id(key)}
                        for value, key in zip(caps.values(), caps.keys())]

    json.dump(coco_format_caps, open(save_name, 'w'))
    # save_json(coco_format_caps, save_name)


def save_json_other_format(caps, save_name):
    format_caps = [{'caption': value, 'image_id': key}
                   for value, key in zip(caps.values(), caps.keys())]

    # save_json(format_caps, save_name)
    json.dump(format_caps, open(save_name, 'w'))
