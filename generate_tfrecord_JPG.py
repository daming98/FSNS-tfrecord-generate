
from random import shuffle
import numpy as np
import glob
import tensorflow as tf
import cv2
import sys
import os
import PIL.Image as Image

#num=str(9)
for n in range(9,53):
    num=str(n)
    image_path = 'data/'+num+'/*.jpg'
    label_path = 'data/'+num+'/*.txt'

    def encode_utf8_string(text, length, dic, null_char_id=20000):
        char_ids_padded = [null_char_id]*length
        char_ids_unpadded = [null_char_id]*len(text)
        for i in range(len(text)):
            if(text[i] in dic):
                hash_id = dic[text[i]]
                char_ids_padded[i] = hash_id
                char_ids_unpadded[i] = hash_id
            else:
                hash_id = 20001
                char_ids_padded[i] = hash_id
                char_ids_unpadded[i] = hash_id
        return char_ids_padded, char_ids_unpadded

    def _bytes_feature(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def _int64_feature(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

    dict={}
    with open('3.txt', encoding="utf") as dict_file:
        for line in dict_file:
            (key, value) = line.strip().split('\t')
            dict[value] = int(key)
    print((dict))

    #image_path = 'data/8/*.jpg'
    addrs_image = glob.glob(image_path)

    #label_path = 'data/8/*.txt'
    addrs_label = glob.glob(label_path)

    print(len(addrs_image))
    print(len(addrs_label))

    name="tfexample_train_"+num+"("+str(len(addrs_image))+")"
    tfrecord_writer  = tf.python_io.TFRecordWriter(name) 
    for j in range(0,int(len(addrs_image))):
        

                # 这是写入操作可视化处理
        print('Train data: {}/{}'.format(j,int(len(addrs_image))))
        sys.stdout.flush()

        img = Image.open(addrs_image[j])

        img = img.resize((50, 500), Image.ANTIALIAS)
        np_data = np.array(img)
        image_data = img.tobytes()
        try:
            for text in open(addrs_label[j], encoding="utf"):
                         char_ids_padded, char_ids_unpadded = encode_utf8_string(
                                    text=text,
                                    dic=dict,
                                    length=37,
                                    )
        except UnicodeDecodeError:
            print(addrs_label[j])



        example = tf.train.Example(features=tf.train.Features(
                            feature={
                                'image/encoded': _bytes_feature(image_data),
                                'image/format': _bytes_feature(b"raw"),
                                'image/width': _int64_feature([np_data.shape[1]]),
                                'image/orig_width': _int64_feature([np_data.shape[1]]),
                                'image/class': _int64_feature(char_ids_padded),
                                'image/unpadded_class': _int64_feature(char_ids_unpadded),
                                'image/text': _bytes_feature(bytes(text, 'utf-8')),
                                # 'height': _int64_feature([crop_data.shape[0]]),
                            }
                        ))
        tfrecord_writer.write(example.SerializeToString())
    tfrecord_writer.close()

    sys.stdout.flush()






