import sys
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QPixmap
from PyQt5 import uic
import dlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import cv2

form_window = uic.loadUiType('./Olove_Old.ui')[0]
second_window = uic.loadUiType('./camerawindow.ui')[0]
class second(QDialog, second_window):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

        self.background_1.setPixmap(QPixmap("./UI/sea.jpg"))
        self.background_1.setScaledContents(True)

        self.background_2.setPixmap(QPixmap("./UI/uni.jpg"))
        self.background_2.setScaledContents(True)

        self.background_3.setPixmap(QPixmap("./UI/club.jpg"))
        self.background_3.setScaledContents(True)

        self.background_4.setPixmap(QPixmap("./UI/merry.jpg"))
        self.background_4.setScaledContents(True)

        self.makeuped.setPixmap(QPixmap("./imgs/makeuped.jpg"))
        self.makeuped.setScaledContents(True)
class Exam(QWidget, form_window):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

        # -이미지 넣기-
        # 타이틀
        self.Olove_Old.setPixmap(QPixmap("./UI/Olove_Old.png"))
        self.Olove_Old.setScaledContents(True)

        # 메이크업 이미지
        self.make_up_1.setPixmap(QPixmap("./UI/디올 썸머룩.PNG"))
        self.make_up_1.setScaledContents(True)

        self.make_up_2.setPixmap(QPixmap("./UI/안유진.PNG"))
        self.make_up_2.setScaledContents(True)

        self.make_up_3.setPixmap(QPixmap("./UI/남자.png"))
        self.make_up_3.setScaledContents(True)

        self.make_up_4.setPixmap(QPixmap("./UI/군인.PNG"))
        self.make_up_4.setScaledContents(True)

        # 장소 이미지
        self.background_1.setPixmap(QPixmap("./UI/sea.jpg"))
        self.background_1.setScaledContents(True)

        self.background_2.setPixmap(QPixmap("./UI/uni.jpg"))
        self.background_2.setScaledContents(True)

        self.background_3.setPixmap(QPixmap("./UI/club.jpg"))
        self.background_3.setScaledContents(True)

        self.background_4.setPixmap(QPixmap("./UI/merry.jpg"))
        self.background_4.setScaledContents(True)

        # 배경
        self.Olove_Old_background.setPixmap(QPixmap("./UI/cosmetics.jpg"))
        self.Olove_Old_background.setScaledContents(True)

        # make_up_n 라벨에 클릭 이벤트 연결
        self.make_up_1.mousePressEvent = self.on_make_up_1_clicked
        self.make_up_2.mousePressEvent = self.on_make_up_2_clicked
        self.make_up_3.mousePressEvent = self.on_make_up_3_clicked
        self.make_up_4.mousePressEvent = self.on_make_up_4_clicked

        # background 라벨을 클릭할때
        self.background_1.mousePressEvent = self.on_background_1_clicked
        self.background_2.mousePressEvent = self.on_background_2_clicked
        self.background_3.mousePressEvent = self.on_background_3_clicked
        self.background_4.mousePressEvent = self.on_background_4_clicked

    # make_up_n 라벨 클릭 시 실행되는 함수
    def on_make_up_1_clicked(self, event):
        print("make_up_1 라벨이 클릭되었습니다.")
        detector = dlib.get_frontal_face_detector()
        shape = dlib.shape_predictor('./models/shape_predictor_68_face_landmarks.dat')

        def align_faces(img):  # 얼굴 정렬하는 함수
            dets = detector(img)
            objs = dlib.full_object_detections()
            for detection in dets:
                s = shape(img, detection)
                objs.append(s)

            faces = dlib.get_face_chips(img, objs, size=256, padding=0.5)
            return faces

        sess = tf.Session()
        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())
        sess.run(init_op)

        saver = tf.train.import_meta_graph('./models/model.meta')
        saver.restore(sess, tf.train.latest_checkpoint('./models'))
        graph = tf.get_default_graph()
        X = graph.get_tensor_by_name('X:0')
        Y = graph.get_tensor_by_name('Y:0')
        Xs = graph.get_tensor_by_name('generator/xs:0')

        def preprocess(img):
            return img / 127.5 - 1

        def deprecess(img):
            return (img + 1) / 2

        img1 = dlib.load_rgb_image('./imgs/makeup.jpg')
        img1_faces = align_faces(img1)

        img2 = dlib.load_rgb_image('./UI/디올 썸머룩.PNG')
        img2_faces = align_faces(img2)

        # fig, axes = plt.subplots(1,2,figsize=(8,5))
        # axes[0].imshow(img1_faces[0])
        # axes[1].imshow(img2_faces[0])
        # plt.show()

        src_img = img1_faces[0]
        ref_img = img2_faces[0]

        X_img = preprocess(src_img)
        X_img = np.expand_dims(X_img, axis=0)

        Y_img = preprocess(ref_img)
        Y_img = np.expand_dims(Y_img, axis=0)

        output = sess.run(Xs, feed_dict={X: X_img, Y: Y_img})
        output_img = deprecess(output[0])

        # plt.savefig('./imgs/makeuped.jpg',output_img)
        output_img_uint8 = (output_img * 255).astype(np.uint8)  # 이미지를 0~255 범위의 정수로 변환
        output_img_bgr = cv2.cvtColor(output_img_uint8, cv2.COLOR_RGB2BGR)  # matplotlib과 cv2의 색 공간 차이를 해결하기 위해 변환

        cv2.imwrite('./imgs/makeuped.jpg', output_img_bgr)  # 이미지 저장
        secondwindow = second()
        secondwindow.exec_()

    def on_make_up_2_clicked(self, event):
        print("make_up_2 라벨이 클릭되었습니다.")

        detector = dlib.get_frontal_face_detector()
        shape = dlib.shape_predictor('./models/shape_predictor_68_face_landmarks.dat')

        def align_faces(img):  # 얼굴 정렬하는 함수
            dets = detector(img)
            objs = dlib.full_object_detections()
            for detection in dets:
                s = shape(img, detection)
                objs.append(s)

            faces = dlib.get_face_chips(img, objs, size=256, padding=0.5)
            return faces

        sess = tf.Session()
        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())
        sess.run(init_op)

        saver = tf.train.import_meta_graph('./models/model.meta')
        saver.restore(sess, tf.train.latest_checkpoint('./models'))
        graph = tf.get_default_graph()
        X = graph.get_tensor_by_name('X:0')
        Y = graph.get_tensor_by_name('Y:0')
        Xs = graph.get_tensor_by_name('generator/xs:0')

        def preprocess(img):
            return img / 127.5 - 1

        def deprecess(img):
            return (img + 1) / 2

        img1 = dlib.load_rgb_image('./imgs/makeup.jpg')
        img1_faces = align_faces(img1)

        img2 = dlib.load_rgb_image('./UI/안유진.PNG')
        img2_faces = align_faces(img2)

        # fig, axes = plt.subplots(1,2,figsize=(8,5))
        # axes[0].imshow(img1_faces[0])
        # axes[1].imshow(img2_faces[0])
        # plt.show()

        src_img = img1_faces[0]
        ref_img = img2_faces[0]

        X_img = preprocess(src_img)
        X_img = np.expand_dims(X_img, axis=0)

        Y_img = preprocess(ref_img)
        Y_img = np.expand_dims(Y_img, axis=0)

        output = sess.run(Xs, feed_dict={X: X_img, Y: Y_img})
        output_img = deprecess(output[0])

        # plt.savefig('./imgs/makeuped.jpg',output_img)
        output_img_uint8 = (output_img * 255).astype(np.uint8)  # 이미지를 0~255 범위의 정수로 변환
        output_img_bgr = cv2.cvtColor(output_img_uint8, cv2.COLOR_RGB2BGR)  # matplotlib과 cv2의 색 공간 차이를 해결하기 위해 변환

        cv2.imwrite('./imgs/makeuped.jpg', output_img_bgr)  # 이미지 저장
        secondwindow = second()
        secondwindow.exec_()
    def on_make_up_3_clicked(self, event):
        print("make_up_3 라벨이 클릭되었습니다.")


        detector = dlib.get_frontal_face_detector()
        shape = dlib.shape_predictor('./models/shape_predictor_68_face_landmarks.dat')

        def align_faces(img):  # 얼굴 정렬하는 함수
            dets = detector(img)
            objs = dlib.full_object_detections()
            for detection in dets:
                s = shape(img, detection)
                objs.append(s)

            faces = dlib.get_face_chips(img, objs, size=256, padding=0.5)
            return faces

        sess = tf.Session()
        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())
        sess.run(init_op)

        saver = tf.train.import_meta_graph('./models/model.meta')
        saver.restore(sess, tf.train.latest_checkpoint('./models'))
        graph = tf.get_default_graph()
        X = graph.get_tensor_by_name('X:0')
        Y = graph.get_tensor_by_name('Y:0')
        Xs = graph.get_tensor_by_name('generator/xs:0')

        def preprocess(img):
            return img / 127.5 - 1

        def deprecess(img):
            return (img + 1) / 2

        img1 = dlib.load_rgb_image('./imgs/makeup.jpg')
        img1_faces = align_faces(img1)

        img2 = dlib.load_rgb_image('./UI/남자.png')
        img2_faces = align_faces(img2)

        # fig, axes = plt.subplots(1,2,figsize=(8,5))
        # axes[0].imshow(img1_faces[0])
        # axes[1].imshow(img2_faces[0])
        # plt.show()

        src_img = img1_faces[0]
        ref_img = img2_faces[0]

        X_img = preprocess(src_img)
        X_img = np.expand_dims(X_img, axis=0)

        Y_img = preprocess(ref_img)
        Y_img = np.expand_dims(Y_img, axis=0)

        output = sess.run(Xs, feed_dict={X: X_img, Y: Y_img})
        output_img = deprecess(output[0])

        # plt.savefig('./imgs/makeuped.jpg',output_img)
        output_img_uint8 = (output_img * 255).astype(np.uint8)  # 이미지를 0~255 범위의 정수로 변환
        output_img_bgr = cv2.cvtColor(output_img_uint8, cv2.COLOR_RGB2BGR)  # matplotlib과 cv2의 색 공간 차이를 해결하기 위해 변환

        cv2.imwrite('./imgs/makeuped.jpg', output_img_bgr)  # 이미지 저장

        secondwindow = second()
        secondwindow.exec_()

    def on_make_up_4_clicked(self, event):
        print("make_up_4 라벨이 클릭되었습니다.")

        detector = dlib.get_frontal_face_detector()
        shape = dlib.shape_predictor('./models/shape_predictor_68_face_landmarks.dat')

        def align_faces(img):  # 얼굴 정렬하는 함수
            dets = detector(img)
            objs = dlib.full_object_detections()
            for detection in dets:
                s = shape(img, detection)
                objs.append(s)

            faces = dlib.get_face_chips(img, objs, size=256, padding=0.5)
            return faces

        sess = tf.Session()
        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())
        sess.run(init_op)

        saver = tf.train.import_meta_graph('./models/model.meta')
        saver.restore(sess, tf.train.latest_checkpoint('./models'))
        graph = tf.get_default_graph()
        X = graph.get_tensor_by_name('X:0')
        Y = graph.get_tensor_by_name('Y:0')
        Xs = graph.get_tensor_by_name('generator/xs:0')

        def preprocess(img):
            return img / 127.5 - 1

        def deprecess(img):
            return (img + 1) / 2

        img1 = dlib.load_rgb_image('./imgs/makeup.jpg')
        img1_faces = align_faces(img1)

        img2 = dlib.load_rgb_image('./UI/군인.PNG')
        img2_faces = align_faces(img2)

        # fig, axes = plt.subplots(1,2,figsize=(8,5))
        # axes[0].imshow(img1_faces[0])
        # axes[1].imshow(img2_faces[0])
        # plt.show()

        src_img = img1_faces[0]
        ref_img = img2_faces[0]

        X_img = preprocess(src_img)
        X_img = np.expand_dims(X_img, axis=0)

        Y_img = preprocess(ref_img)
        Y_img = np.expand_dims(Y_img, axis=0)

        output = sess.run(Xs, feed_dict={X: X_img, Y: Y_img})
        output_img = deprecess(output[0])

        # plt.savefig('./imgs/makeuped.jpg',output_img)
        output_img_uint8 = (output_img * 255).astype(np.uint8)  # 이미지를 0~255 범위의 정수로 변환
        output_img_bgr = cv2.cvtColor(output_img_uint8, cv2.COLOR_RGB2BGR)  # matplotlib과 cv2의 색 공간 차이를 해결하기 위해 변환

        cv2.imwrite('./imgs/makeuped.jpg', output_img_bgr)  # 이미지 저장

        secondwindow = second()
        secondwindow.exec_()

    # background_n 라벨 클릭 시 실행되는 함수
    def on_background_1_clicked(self, event):
        print("background_1 라벨이 클릭되었습니다.")
    def on_background_2_clicked(self, event):
        print("background_2 라벨이 클릭되었습니다.")
    def on_background_3_clicked(self, event):
        print("background_3 라벨이 클릭되었습니다.")
    def on_background_4_clicked(self, event):
        print("background_4 라벨이 클릭되었습니다.")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    mainWindow = Exam()
    mainWindow.show()
    sys.exit(app.exec_())