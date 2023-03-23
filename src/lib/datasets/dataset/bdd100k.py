from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from re import A

import pycocotools.coco as coco
import numpy as np
import json
import os

import torch.utils.data as data
import tools.det_eval.kitti_common as kitti
from tools.det_eval.eval import get_official_eval_result

def _read_imageset_file(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    return [int(line) for line in lines]

class BDD100K(data.Dataset):
  num_classes = 3
  default_resolution = [512, 896]
  mean = np.array([0.485, 0.456, 0.406], np.float32).reshape(1, 1, 3)
  std = np.array([0.229, 0.224, 0.225], np.float32).reshape(1, 1, 3)

  def __init__(self, opt, split):
    super(BDD100K, self).__init__()
    self.data_dir = os.path.join(opt.data_dir, 'bdd100k')
    self.img_dir = os.path.join(self.data_dir, f'images/100k/{split}' if split != 'test' else 'images/100k/val')
    if split == 'test':
      self.annot_path = os.path.join(self.data_dir, 'labels', 'val.json')
    else:
      self.annot_path = os.path.join(self.data_dir, 'labels', '{}.json').format(split)
    self.max_objs = 128
    self.class_name = [
      '__background__', 'pedestrian', 'car', 'rider']
    self.cat_ids = {
      1: 0, 2: 2, 3: 1, 4: -3, 5: -3, 6: -3, 7: -3, 8: -3, 9: -3, 10: -3
    }
    self._data_rng = np.random.RandomState(123)
    self._eig_val = np.array([0.2141788, 0.01817699, 0.00341571],
                             dtype=np.float32)
    self._eig_vec = np.array([
        [-0.58752847, -0.69563484, 0.41340352],
        [-0.5832747, 0.00994535, -0.81221408],
        [-0.56089297, 0.71832671, 0.41158938]
    ], dtype=np.float32)
    self.split = split
    self.opt = opt

    print('==> initializing BDD100K {} data.'.format(split))
    self.coco = coco.COCO(self.annot_path)
    self.images = self.coco.getImgIds()
    self.num_samples = len(self.images)

    print('Loaded {} {} samples'.format(split, self.num_samples))

  def _to_float(self, x):
    return float("{:.2f}".format(x))

  def convert_eval_format(self, all_bboxes):
    # import pdb; pdb.set_trace()
    detections = []
    for image_id in all_bboxes:
      for cls_ind in all_bboxes[image_id]:
        category_id = (cls_ind - 1) - 1
        for bbox in all_bboxes[image_id][cls_ind]:
          bbox[2] -= bbox[0]
          bbox[3] -= bbox[1]
          score = bbox[4]
          bbox_out  = list(map(self._to_float, bbox[0:4]))

          detection = {
              "image_id": int(image_id),
              "category_id": int(category_id),
              "bbox": bbox_out,
              "score": float("{:.2f}".format(score))
          }
          if len(bbox) > 5:
              extreme_points = list(map(self._to_float, bbox[5:13]))
              detection["extreme_points"] = extreme_points
          detections.append(detection)
    return detections

  def __len__(self):
    return self.num_samples

  def save_results(self, results, results_dir):
    if not os.path.exists(results_dir):
      os.makedirs(results_dir)
    for img_id in results.keys():
      out_path = os.path.join(results_dir, '{:06d}.txt'.format(img_id))
      f = open(out_path, 'w')
      for cls_ind in results[img_id]:
        for j in range(len(results[img_id][cls_ind])):
          class_name = self.class_name[cls_ind]
          f.write('{} 0.0 0 -10'.format(class_name))
          for i in range(len(results[img_id][cls_ind][j])-1):
            f.write(' {:.2f}'.format(results[img_id][cls_ind][j][i]))
          f.write(' 0.0 0.0 0.0 0.0 0.0 0.0 0.0 {:.2f}'.format(results[img_id][cls_ind][j][i+1]))
          f.write('\n')
      f.close()
    
  def convert_labels_to_kitti_format(self, gt_path, gt_json_path):
    if not os.path.exists(gt_path):
      print("Converting labels to kitti format...")
      os.mkdir(gt_path)

      with open(gt_json_path, 'r') as f:
        gt_file = json.load(f)
      gt_cat_names = [c['name'] for c in gt_file['categories']]
      gt_raw_annos = gt_file['annotations']

      for gt_ann in gt_raw_annos:
        img_id = gt_ann['image_id']
        out_path = os.path.join(gt_path, '{:06d}.txt'.format(img_id))
        with open(out_path, "a") as f:
          class_name = gt_cat_names[gt_ann['category_id'] - 1].replace(' ', '')
          f.write('{} 0.0 0 -10'.format(class_name))
          bbox = gt_ann['bbox']
          bbox[2] += bbox[0]
          bbox[3] += bbox[1]
          for i in range(len(bbox)):
            f.write(' {:.2f}'.format(bbox[i]))
          f.write(' 0.0 0.0 0.0 0.0 0.0 0.0 0.0')
          f.write('\n')
  
  def run_eval(self, results, save_dir):
    results_dir = os.path.join(save_dir, f'results/{self.opt.dataset}')
    self.save_results(results, results_dir)

    det_path = results_dir
    dt_annos = kitti.get_label_annos(det_path)
    gt_path = os.path.join(self.data_dir, 'labels/labels')
    gt_json_path = os.path.join(gt_path, '../val.json')
    self.convert_labels_to_kitti_format(gt_path, gt_json_path)
    gt_annos = kitti.get_label_annos(gt_path)
    print(get_official_eval_result(gt_annos, dt_annos, ['car'], self.opt.dataset))
