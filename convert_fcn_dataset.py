#!/usr/bin/env python3
import logging
import os
import io
import hashlib
import re

import cv2
import numpy as np
import tensorflow as tf
from vgg import vgg_16

import PIL.Image

from lxml import etree
from object_detection.utils import dataset_util
from object_detection.utils import label_map_util

flags = tf.app.flags
flags.DEFINE_string('data_dir', '', 'Root directory to raw pet dataset.')
flags.DEFINE_string('output_dir', '', 'Path to directory to output TFRecords.')

FLAGS = flags.FLAGS

classes = ['background', 'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
           'dog', 'horse', 'motorbike', 'person', 'potted plant',
           'sheep', 'sofa', 'train', 'tv/monitor']
# RGB color for each class
colormap = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128],
            [128, 0, 128], [0, 128, 128], [
                128, 128, 128], [64, 0, 0], [192, 0, 0],
            [64, 128, 0], [192, 128, 0], [64, 0, 128], [192, 0, 128],
            [64, 128, 128], [192, 128, 128], [0, 64, 0], [128, 64, 0],
            [0, 192, 0], [128, 192, 0], [0, 64, 128]]


cm2lbl = np.zeros(256**3)
for i, cm in enumerate(colormap):
    cm2lbl[(cm[0] * 256 + cm[1]) * 256 + cm[2]] = i


def image2label(im):
    data = im.astype('int32')
    # cv2.imread. default channel layout is BGR
    idx = (data[:, :, 2] * 256 + data[:, :, 1]) * 256 + data[:, :, 0]
    return np.array(cm2lbl[idx])


def dict_to_tf_example(data, label):
    with open(data, 'rb') as inf:
        encoded_data = inf.read()
    img_label = cv2.imread(label)
    img_mask = image2label(img_label)
    encoded_label = img_mask.astype(np.uint8).tobytes()

    print('data=', data)
    print('os.path.basename(data)=', os.path.basename(data))
#     print('label=', label)
#     print('encoded_data=', encoded_data)
#     print('img_label=', img_label)
#     print('img_mask=', img_mask)
#     print('encoded_label=', encoded_label)
    
    height, width = img_label.shape[0], img_label.shape[1]
    if height < vgg_16.default_image_size or width < vgg_16.default_image_size:
        # 保证最后随机裁剪的尺寸
        return None

    # Your code here, fill the dict
    feature_dict = {
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(os.path.basename(data).encode('utf8')),
        'image/encoded': dataset_util.bytes_feature(encoded_data),
        'image/label': dataset_util.bytes_feature(encoded_label),
        'image/format': dataset_util.bytes_feature('jpeg'.encode('utf8')),
    }
    
#     print('feature_dict=',feature_dict)
#     print('feature_dict.type=',feature_dict.type)
    example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
    return example

# def get_class_name_from_filename(file_name):
#   """Gets the class name from a file.

#   Args:
#     file_name: The file name to get the class name from.
#                ie. "american_pit_bull_terrier_105.jpg"

#   Returns:
#     A string of the class name.
#   """
#   print('file_name=', file_name)

#   match = re.match(r'([A-Za-z_]+)(_[0-9]+\.jpg)', file_name, re.I)
#   print('match=', match)
#   return match.groups()[0]



def dict_to_tf_example_re(data,
                       mask_path,
                       image_subdirectory,
                       data_path,
                       label_path,
                       ignore_difficult_instances=False,
                       ):
  """Convert XML derived dict to tf.Example proto.
  
  Notice that this function normalizes the bounding box coordinates provided
  by the raw data.

  Args:
    data: dict holding PASCAL XML fields for a single image (obtained by
      running dataset_util.recursive_parse_xml_to_dict)
    mask_path: String path to PNG encoded mask.
    label_map_dict: A map from string label names to integers ids.
    image_subdirectory: String specifying subdirectory within the
      Pascal dataset directory holding the actual image data.
    ignore_difficult_instances: Whether to skip difficult instances in the
      dataset  (default: False).
    faces_only: If True, generates bounding boxes for pet faces.  Otherwise
      generates bounding boxes (as well as segmentations for full pet bodies).

  Returns:
    example: The converted tf.Example.

  Raises:
    ValueError: if the image pointed to by data['filename'] is not a valid JPEG
  """
#   data_path, label_path, xml_path = zip(*file_pars)
  
#   lens = len(data_path)
#   print('lens=', lens)

#   for idx in range(lens):  
#       dict_to_tf_example(data_path[idx], label_path[idx])

#   print('data_path=', data_path)
 
  return dict_to_tf_example(data_path, label_path)
#   with open(data_path, 'rb') as inf:
#     encoded_data = inf.read()
#   img_label = cv2.imread(label_path)
#   img_mask = image2label(img_label)
#   encoded_label = img_mask.astype(np.uint8).tobytes()

#     #     print('data=', data)
#     #     print('label=', label)
#     #     print('encoded_data=', encoded_data)
# #   print('img_label=', img_label)
# #   print('img_mask=', img_mask)
#     #     print('encoded_label=', encoded_label)

#   height, width = img_label.shape[0], img_label.shape[1]
#   if height < vgg_16.default_image_size or width < vgg_16.default_image_size:
#         # 保证最后随机裁剪的尺寸
#       return None
    


# #   img_path = os.path.join(image_subdirectory, data['filename'])
#   with tf.gfile.GFile(data_path, 'rb') as fid:
#     encoded_jpg = fid.read()
#   encoded_jpg_io = io.BytesIO(encoded_jpg)
#   image = PIL.Image.open(encoded_jpg_io)
#   if image.format != 'JPEG':
#     raise ValueError('Image format not JPEG')
#   key = hashlib.sha256(encoded_jpg).hexdigest()

#   with tf.gfile.GFile(mask_path, 'rb') as fid:
#     encoded_mask_png = fid.read()
#   encoded_png_io = io.BytesIO(encoded_mask_png)
#   mask = PIL.Image.open(encoded_png_io)
#   if mask.format != 'PNG':
#     raise ValueError('Mask format not PNG')

#   mask_np = np.asarray(mask)
#   nonbackground_indices_x = np.any(mask_np != 2, axis=0)
#   nonbackground_indices_y = np.any(mask_np != 2, axis=1)
#   nonzero_x_indices = np.where(nonbackground_indices_x)
#   nonzero_y_indices = np.where(nonbackground_indices_y)

#   width = int(data['size']['width'])
#   height = int(data['size']['height'])

#   xmins = []
#   ymins = []
#   xmaxs = []
#   ymaxs = []
#   classes_tmp = []
# #   classes = classes

#   classes_text = []
#   truncated = []
#   poses = []
#   difficult_obj = []
#   masks = []
#   for obj in data['object']:
#     print('obj=', obj)
    
#     difficult = bool(int(obj['difficult']))
#     if ignore_difficult_instances and difficult:
#       continue
#     difficult_obj.append(int(difficult))
   
#     xmin = float(np.min(nonzero_x_indices))
#     xmax = float(np.max(nonzero_x_indices))
#     ymin = float(np.min(nonzero_y_indices))
#     ymax = float(np.max(nonzero_y_indices))

#     xmins.append(xmin / width)
#     ymins.append(ymin / height)
#     xmaxs.append(xmax / width)
#     ymaxs.append(ymax / height)
# #     class_name = get_class_name_from_filename(data['filename'])
# #     print('class_name=',class_name)
# #     classes_text.append(class_name.encode('utf8'))
# #     classes_tmp.append(classes[class_name])
#     truncated.append(int(obj['truncated']))
#     poses.append(obj['pose'].encode('utf8'))
# #     if not faces_only:
#     mask_remapped = mask_np != 2
#     masks.append(mask_remapped)

#   feature_dict = {
#       'image/height': dataset_util.int64_feature(height),
#       'image/width': dataset_util.int64_feature(width),
#       'image/filename': dataset_util.bytes_feature(
#           data['filename'].encode('utf8')),
#       'image/source_id': dataset_util.bytes_feature(
#           data['filename'].encode('utf8')),
#       'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
#       'image/encoded': dataset_util.bytes_feature(encoded_jpg),
#       'image/format': dataset_util.bytes_feature('jpeg'.encode('utf8')),
#       'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
#       'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
#       'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
#       'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
# #       'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
# #       'image/object/class/label': dataset_util.int64_list_feature(classes_tmp),
#       'image/object/difficult': dataset_util.int64_list_feature(difficult_obj),
#       'image/object/truncated': dataset_util.int64_list_feature(truncated),
#       'image/object/view': dataset_util.bytes_list_feature(poses),
#   }
# #   if not faces_only:
#   mask_stack = np.stack(masks).astype(np.float32)
#   masks_flattened = np.reshape(mask_stack, [-1])
#   feature_dict['image/object/mask'] = (
#     dataset_util.float_list_feature(masks_flattened.tolist()))

#   example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
#   return example


def create_tf_record_re(output_filename,
#                      label_map_dict,
                     annotations_dir,
                     image_dir,
                     examples,file_pars
                       ):
  """Creates a TFRecord file from examples.

  Args:
    output_filename: Path to where output file is saved.
    label_map_dict: The label map dictionary.
    annotations_dir: Directory where annotation files are stored.
    image_dir: Directory where image files are stored.
    examples: Examples to parse and save to tf record.
    faces_only: If True, generates bounding boxes for pet faces.  Otherwise
      generates bounding boxes (as well as segmentations for full pet bodies).
  """
  writer = tf.python_io.TFRecordWriter(output_filename)
  data_path, label_path, xml_path = zip(*file_pars)
  examples = xml_path
  masks = label_path
#   print('masks=',masks)
#   for idx, example in enumerate(examples):
  lens = len(data_path)
  print('lens=', lens)

  for idx in range(lens):
    print('idx=',idx)
    if idx % 100 == 0:
      logging.info('On image %d of %d', idx, len(examples))
    
#     print('annotations_dir=', annotations_dir)
    
#     xml_path = os.path.join(annotations_dir, example + '.xml')
    xml_path = examples[idx]
#     mask_path = os.path.join(annotations_dir, 'trimaps', example + '.png')
    mask_path = masks[idx]
    if not os.path.exists(xml_path):
      logging.warning('Could not find %s, ignoring example.', xml_path)
      continue
    with tf.gfile.GFile(xml_path, 'r') as fid:
      xml_str = fid.read()
    xml = etree.fromstring(xml_str)
    data = dataset_util.recursive_parse_xml_to_dict(xml)['annotation']

    try:
        tf_example = dict_to_tf_example_re(
          data, mask_path,
          image_dir, data_path[idx], label_path[idx])
        writer.write(tf_example.SerializeToString())
#     except ValueError:
    except Exception as e:
      logging.warning('Invalid example: %s, ignoring.%s', xml_path,e)

  writer.close()


def create_tf_record(output_filename, file_pars):
  # Your code here
#   label_map_dict = label_map_util.get_label_map_dict(FLAGS.label_map_path)
  data_dir = '/root/VOCdevkit/VOC2012/'
  annotations_dir = os.path.join(data_dir, 'Annotations')
#   examples_path = os.path.join(annotations_dir, 'trainval.txt')
#   examples_list = dataset_util.read_examples_list(examples_path)
#   data, label = zip(*file_pars)
#   examples_list = data
  examples_list = []
  image_dir = os.path.join(data_dir, 'images')
  create_tf_record_re(output_filename,
#                      label_map_dict,
                     annotations_dir,
                     image_dir,
                     examples_list,file_pars)
                        

def read_images_names(root, train=True):
    txt_fname = os.path.join(root, 'ImageSets/Segmentation/', 'train.txt' if train else 'val.txt')

    with open(txt_fname, 'r') as f:
        images = f.read().split()

    data = []
    label = []
    xml = []
    
    for fname in images:
        data.append('%s/JPEGImages/%s.jpg' % (root, fname))
        label.append('%s/SegmentationClass/%s.png' % (root, fname))
        xml.append('%s/Annotations/%s.xml' % (root, fname))
#         print('data=',data)
#         print('label=',label)

    return zip(data, label, xml)


# def read_classes_names(root, train=True):
#     txt_fname = os.path.join(root, 'ImageSets/Main/')

#     list_tmp = os.listdir(txt_fname) #列出文件夹下所有的目录与文件
#     for i in range(0,len(list_tmp)):
#       path = os.path.join(txt_fname,list_tmp[i])
#       if os.path.isfile(path):
#         with open(path, 'r') as f:
#           images = f.read().split()
#           print('images=',images)

#     data = []
#     label = []
#     xml = []
    
#     for fname in images:
#         data.append('%s/JPEGImages/%s.jpg' % (root, fname))
#         label.append('%s/SegmentationClass/%s.png' % (root, fname))
#         xml.append('%s/Annotations/%s.xml' % (root, fname))
# #         print('data=',data)
# #         print('label=',label)

#     return zip(data, label, xml)


def main(_):
    logging.info('Prepare dataset file names')

    train_output_path = os.path.join(FLAGS.output_dir, 'fcn_train.record')
    val_output_path = os.path.join(FLAGS.output_dir, 'fcn_val.record')

    train_files = read_images_names(FLAGS.data_dir, True)
    val_files = read_images_names(FLAGS.data_dir, False)
    print('train_files=',train_files)
    print('val_files=',val_files)

    create_tf_record(train_output_path, train_files)
    create_tf_record(val_output_path, val_files)


if __name__ == '__main__':
    tf.app.run()
