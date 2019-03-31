import os 
import tensorflow as tf 
from PIL import Image #image后面需要使用 

cwd = '/home/cz/Daming/FSNS-tfrecord-generate-master/1/'#手动输入路径 
filename_queue = tf.train.string_input_producer(["tfexample_test_46(1244)"]) #读入流中 
reader = tf.TFRecordReader() 
_, serialized_example = reader.read(filename_queue) #返回文件名和文件 
features = tf.parse_single_example(serialized_example, 
                                    features={ 
                                        'image/text': tf.FixedLenFeature([], tf.string), 
                                        'image/encoded' : tf.FixedLenFeature([], tf.string), 
                                    }) #取出包含image和label的feature对象 
image = tf.decode_raw(features['image/encoded'], tf.uint8) 
image = tf.reshape(image, [250, 27, 3]) 
label = tf.cast(features['image/text'], tf.string) 
with tf.Session() as sess: #开始一个会话 
    init_op = tf.global_variables_initializer() 
    sess.run(init_op) 
    coord=tf.train.Coordinator() 
    threads= tf.train.start_queue_runners(coord=coord) 
    for i in range(20): 
        example, l = sess.run([image,label])#在会话中取出image和label 
        img=Image.fromarray(example, 'RGB')#这里Image是之前提到的 
        li=l.decode('utf-8')
        img.save(cwd+str(i)+'_''Label_'+str(li)+'.jpg')#存下图片 
        print('----------------------------') 
        print(example, l) 
    coord.request_stop() 
    coord.join(threads)
