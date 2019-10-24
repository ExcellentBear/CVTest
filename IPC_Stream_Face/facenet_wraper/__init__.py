# -*- coding:utf-8 -*-
"""
性能瓶颈在MTCNN，故QUEUE作为进程间通讯介质可行
若图片较小，机器性能高，MTCNN很快的话，可以采用RawArray等
甚至自己基于mmp共享内存封装其它效率较高的进程通讯介质
"""
from PIL import Image
from facenet.src.align import detect_face
from facenet.src import facenet
import numpy as np
import tensorflow as tf
import os
from multiprocessing import Process, Queue

__all__ = ["Recon"]


class MtCNN(Process):
    def __init__(self, process_queue_in: Queue, process_queue_out: Queue):
        super().__init__()
        self.inq = process_queue_in
        self.out_q = process_queue_out

    def run(self):
        minsize = 20  # minimum size of face
        threshold = [0.6, 0.7, 0.7]  # three steps's threshold
        factor = 0.709  # scale factor
        margin = 44
        image_size = 160
        gpu_memory_fraction = 1.0
        with tf.Graph().as_default():
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
            sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
            with sess.as_default():
                p_net, r_net, o_net = detect_face.create_mtcnn(sess, None)
        while True:
            img = self.inq.get()
            img_size = np.asarray(img.shape)[0:2]
            bounding_boxes, _ = detect_face.detect_face(img, minsize, p_net, r_net, o_net, threshold, factor)
            src = img.copy()
            dist_white_ends = []
            for num in range(bounding_boxes.shape[0]):
                det = np.squeeze(bounding_boxes[num, 0:4])
                bb = np.zeros(4, dtype=np.int32)
                bb[0] = np.maximum(det[0] - margin / 2, 0)
                bb[1] = np.maximum(det[1] - margin / 2, 0)
                bb[2] = np.minimum(det[2] + margin / 2, img_size[1])
                bb[3] = np.minimum(det[3] + margin / 2, img_size[0])
                cropped = img[bb[1]:bb[3], bb[0]:bb[2], :]

                if (bb[0] >= 0) & (bb[0] < src.shape[1]):
                    src[bb[1]:bb[3], bb[0], :] = 255
                else:
                    src[bb[1]:bb[3], src.shape[1]-1, :] = 255

                if (bb[2] >= 0) & (bb[2] < src.shape[1]):
                    src[bb[1]:bb[3], bb[2], :] = 255
                else:
                    src[bb[1]:bb[3], src.shape[1]-1, :] = 255

                if (bb[1] >= 0) & (bb[1] < src.shape[0]):
                    src[bb[1], bb[0]:bb[2], :] = 255
                else:
                    src[src.shape[0]-1, bb[0]:bb[2], :] = 255

                if (bb[3] >= 0) & (bb[3] < src.shape[0]):
                    src[bb[3], bb[0]:bb[2], :] = 255
                else:
                    src[src.shape[0] - 1, bb[0]:bb[2], :] = 255

                pil_im = Image.fromarray(cropped)
                aligned = pil_im.resize((image_size, image_size), Image.BILINEAR)
                aligned = np.array(aligned)
                pre_whitened = facenet.prewhiten(aligned)
                dist_white_ends.append(pre_whitened)
            self.out_q.put({"src": src, "dst": dist_white_ends})


class EmbeddingCmp(Process):
    def __init__(self, mt_i_queue: Queue, mt_o_queue: Queue, process_queue_out: Queue,
                 to_prepare_images: str, model_path: str):
        super().__init__()
        self.fd_q = mt_i_queue
        self.i_q = mt_o_queue
        self.o_q = process_queue_out
        self.key_value = {}
        self.p_path = to_prepare_images
        self.model_path = model_path
        files = os.listdir(self.p_path)
        abs_paths = [os.path.join(self.p_path, x) for x in files]
        for i in range(abs_paths.__len__()):
            name = abs_paths[i]
            image = Image.open(name)
            image = image.convert("RGB")
            image = np.array(image)
            self.fd_q.put(image)
            dist_white_ends_dict = self.i_q.get()
            dist_white_ends = dist_white_ends_dict["dst"]
            if dist_white_ends:
                self.key_value[files[i]] = dist_white_ends[0][None]

    def run(self):
        with tf.Graph().as_default():
            with tf.Session() as sess:
                facenet.load_model(self.model_path)
                images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
                embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
                phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
                for k in self.key_value.keys():
                    feed_dict = {images_placeholder: self.key_value[k], phase_train_placeholder: False}
                    emb0 = sess.run(embeddings, feed_dict=feed_dict)
                    self.key_value[k] = emb0

                while True:
                    image_dict = self.i_q.get()
                    dist_white_ends = image_dict["dst"]
                    label_distance = []
                    for white_end in dist_white_ends:
                        label = ""
                        f_max = np.Inf
                        image = white_end[None]
                        feed_dict = {images_placeholder: image, phase_train_placeholder: False}
                        emb = sess.run(embeddings, feed_dict=feed_dict)
                        for k, v in self.key_value.items():
                            distance = np.sqrt(np.sum(np.square(np.subtract(emb[0:], v[0:]))))
                            if distance < f_max:
                                f_max = distance
                                label = k
                        label_distance.append((label, f_max))
                    image_dict["ret"] = label_distance
                    self.o_q.put(image_dict)


root_dir = os.path.dirname(os.path.realpath(__file__))
model_dir_ = os.path.join(root_dir, "../open_source_models").replace("\\", "/")
label_dir_ = os.path.join(root_dir, "../png_for_compare").replace("\\", "/")


class Recon:
    def __init__(self, model_dir=model_dir_, label_dir=label_dir_):
        self.feed = Queue()
        self.mid = Queue()
        self.out = Queue()
        t = MtCNN(self.feed, self.mid)
        t.start()
        t1 = EmbeddingCmp(self.feed, self.mid, self.out, label_dir, model_dir)
        t1.start()

    def face_check(self, _im: np.ndarray):
        self.feed.put(_im)
        return self.out.get()
