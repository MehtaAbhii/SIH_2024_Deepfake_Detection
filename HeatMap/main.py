from imutils.video import VideoStream
from imutils import face_utils
import imutils
import dlib
import cv2

from tensorflow.keras.models import Model
import tensorflow as tf
import numpy as np
import cv2
import argparse
import imutils

from classifiers import *
import numpy as np

import datetime
import time

# -*- coding:utf-8 -*-

from tensorflow.keras.models import Model as KerasModel
from tensorflow.keras.layers import Input, Dense, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Dropout, Reshape, Concatenate, LeakyReLU
from tensorflow.keras.optimizers import Adam




IMGWIDTH = 256

class Classifier:
    def __init__():
        self.model = 0
    
    def predict(self, x):
        if x.size == 0:
            return []
        return self.model.predict(x)
    
    def fit(self, x, y):
        return self.model.train_on_batch(x, y)
    
    def get_accuracy(self, x, y):
        return self.model.test_on_batch(x, y)
    
    def load(self, path):
        self.model.load_weights(path)


class Meso1(Classifier):
    """
    Feature extraction + Classification
    """
    def __init__(self, learning_rate = 0.001, dl_rate = 1):
        self.model = self.init_model(dl_rate)
        optimizer = Adam(lr = learning_rate)
        self.model.compile(optimizer = optimizer, loss = 'mean_squared_error', metrics = ['accuracy'])
    
    def init_model(self, dl_rate):
        x = Input(shape = (IMGWIDTH, IMGWIDTH, 3))
        
        x1 = Conv2D(16, (3, 3), dilation_rate = dl_rate, strides = 1, padding='same', activation = 'relu')(x)
        x1 = Conv2D(4, (1, 1), padding='same', activation = 'relu')(x1)
        x1 = BatchNormalization()(x1)
        x1 = MaxPooling2D(pool_size=(8, 8), padding='same')(x1)

        y = Flatten()(x1)
        y = Dropout(0.5)(y)
        y = Dense(1, activation = 'sigmoid')(y)
        return KerasModel(inputs = x, outputs = y)


class Meso4(Classifier):
    def __init__(self, learning_rate = 0.001):
        self.model = self.init_model()
        optimizer = Adam(lr = learning_rate)
        self.model.compile(optimizer = optimizer, loss = 'mean_squared_error', metrics = ['accuracy'])
    
    def init_model(self): 
        x = Input(shape = (IMGWIDTH, IMGWIDTH, 3))
        
        x1 = Conv2D(8, (3, 3), padding='same', activation = 'relu')(x)
        x1 = BatchNormalization()(x1)
        x1 = MaxPooling2D(pool_size=(2, 2), padding='same')(x1)
        
        x2 = Conv2D(8, (5, 5), padding='same', activation = 'relu')(x1)
        x2 = BatchNormalization()(x2)
        x2 = MaxPooling2D(pool_size=(2, 2), padding='same')(x2)
        
        x3 = Conv2D(16, (5, 5), padding='same', activation = 'relu')(x2)
        x3 = BatchNormalization()(x3)
        x3 = MaxPooling2D(pool_size=(2, 2), padding='same')(x3)
        
        x4 = Conv2D(16, (5, 5), padding='same', activation = 'relu')(x3)
        x4 = BatchNormalization()(x4)
        x4 = MaxPooling2D(pool_size=(4, 4), padding='same')(x4)
        
        y = Flatten()(x4)
        y = Dropout(0.5)(y)
        y = Dense(16)(y)
        y = LeakyReLU(alpha=0.1)(y)
        y = Dropout(0.5)(y)
        y = Dense(1, activation = 'sigmoid')(y)

        return KerasModel(inputs = x, outputs = y)


class MesoInception4(Classifier):
    def __init__(self, learning_rate = 0.001):
        self.model = self.init_model()
        optimizer = Adam(lr = learning_rate)
        self.model.compile(optimizer = optimizer, loss = 'mean_squared_error', metrics = ['accuracy'])
    
    def InceptionLayer(self, a, b, c, d):
        def func(x):
            x1 = Conv2D(a, (1, 1), padding='same', activation='relu')(x)
            
            x2 = Conv2D(b, (1, 1), padding='same', activation='relu')(x)
            x2 = Conv2D(b, (3, 3), padding='same', activation='relu')(x2)
            
            x3 = Conv2D(c, (1, 1), padding='same', activation='relu')(x)
            x3 = Conv2D(c, (3, 3), dilation_rate = 2, strides = 1, padding='same', activation='relu')(x3)
            
            x4 = Conv2D(d, (1, 1), padding='same', activation='relu')(x)
            x4 = Conv2D(d, (3, 3), dilation_rate = 3, strides = 1, padding='same', activation='relu')(x4)

            y = Concatenate(axis = -1)([x1, x2, x3, x4])
            
            return y
        return func
    
    def init_model(self):
        x = Input(shape = (IMGWIDTH, IMGWIDTH, 3))
        
        x1 = self.InceptionLayer(1, 4, 4, 2)(x)
        x1 = BatchNormalization()(x1)
        x1 = MaxPooling2D(pool_size=(2, 2), padding='same')(x1)
        
        x2 = self.InceptionLayer(2, 4, 4, 2)(x1)
        x2 = BatchNormalization()(x2)
        x2 = MaxPooling2D(pool_size=(2, 2), padding='same')(x2)        
        
        x3 = Conv2D(16, (5, 5), padding='same', activation = 'relu')(x2)
        x3 = BatchNormalization()(x3)
        x3 = MaxPooling2D(pool_size=(2, 2), padding='same')(x3)
        
        x4 = Conv2D(16, (5, 5), padding='same', activation = 'relu')(x3)
        x4 = BatchNormalization()(x4)
        x4 = MaxPooling2D(pool_size=(4, 4), padding='same')(x4)
        
        y = Flatten()(x4)
        y = Dropout(0.5)(y)
        y = Dense(16)(y)
        y = LeakyReLU(alpha=0.1)(y)
        y = Dropout(0.5)(y)
        y = Dense(1, activation = 'sigmoid')(y)

        return KerasModel(inputs = x, outputs = y)
	
class GradCAM:
	def __init__(self, model, classIdx, layerName=None):
		self.model = model
		self.classIdx = classIdx
		self.layerName = layerName
		if self.layerName is None:
			self.layerName = self.find_layer()

	def find_layer(self):
		for layer in reversed(self.model.layers):
			if len(layer.output_shape) == 4 and not any(x in layer.name for x in ["pool", "batch"]):
				return layer.name
		raise ValueError("Could not find layer, cannot apply CAM.")

	def overlay_heatmap(self, heatmap, image, alpha=0.5,
		colormap=cv2.COLORMAP_JET):
		heatmap = cv2.applyColorMap(heatmap, colormap)
		output = cv2.addWeighted(image, alpha, heatmap, 1 - alpha, 0)
		return (heatmap, output)

	def compute_heatmap(self, image, eps=1e-8):
		gradModel = Model(
			inputs=[self.model.inputs[0]],
			outputs=[self.model.get_layer(self.layerName).output, self.model.output])
		with tf.GradientTape() as tape:
			inputs = tf.cast(image, tf.float32)
			(convOutputs, predictions) = gradModel(inputs)
			loss = predictions[:, self.classIdx]
		grads = tape.gradient(loss, convOutputs)
		castConvOutputs = tf.cast(convOutputs > 0, "float32")
		castGrads = tf.cast(grads > 0, "float32")
		guidedGrads = castConvOutputs * castGrads * grads
		convOutputs = convOutputs[0]
		guidedGrads = guidedGrads[0]
		weights = tf.reduce_mean(guidedGrads, axis=(0, 1))
		cam = tf.reduce_sum(tf.multiply(weights, convOutputs), axis=-1)
		(w, h) = (image.shape[2], image.shape[1])
		heatmap = cv2.resize(cam.numpy(), (w, h))
		numer = heatmap - np.min(heatmap)
		denom = (heatmap.max() - heatmap.min()) + eps
		heatmap = numer / denom
		heatmap = (heatmap * 255).astype("uint8")
		return heatmap


def get_heatmap_detection(model, in_imgg):

	orig = in_imgg
	img_rgb = np.copy(orig)
	img_rgb = cv2.resize(img_rgb,(256,256))  # resize
	img_rgb = np.array(img_rgb).astype(np.float32)/255.0  # scaling
	img_rgb = np.expand_dims(img_rgb, axis=0)  # expand dimension

	preds = model.predict(img_rgb)
	i = np.argmax(preds[0])
	# initialize our gradient class activation map and build the heatmap
	cam = GradCAM(model, i)
	heatmap = cam.compute_heatmap(img_rgb)
	# resize the resulting heatmap to the original input image dimensions
	# and then overlay heatmap on top of the image
	heatmap = cv2.resize(heatmap, (orig.shape[1], orig.shape[0]))
	(heatmap, output) = cam.overlay_heatmap(heatmap, orig, alpha=0.5)
	return output



class VideoGanAnalyzer:

	def __init__(self, name_of_video, model_path, is_gan_threshold=0.5):
		self.name_of_video = name_of_video
		self.model_path = model_path
		self.is_gan_threshold = is_gan_threshold
		self.detector = None
		self.vs = None
		self.fps = None
		self.writer = None
		self.H = None
		self.W = None
		self.model_input_shape = (256, 256)
		self.list_frames = []
		self.list_of_faces = []
		self.classifier = None

	def load_face_detector(self):
		print("[INFO] loading frontal face detector...")
		self.detector = dlib.get_frontal_face_detector()

	def load_video(self):
		print("[INFO] loading video...")
		self.vs = cv2.VideoCapture(self.name_of_video)
		self.fps = self.vs.get(cv2.CAP_PROP_FPS)
		print("[INFO] fps : {0}".format(self.fps))
		time.sleep(2.0)

	def add_face_to_list(self, face_image, id_frame, id_face, coord):
		"""{"image" : np.array, "id_frame" : int, "coord" : [x, y, w, h]}"""
		return {"image" : face_image, "image_resized" : cv2.resize(np.copy(face_image), self.model_input_shape),
				"id_frame" : id_frame, "id_face" : id_face, "coord" : coord}

	def read_video(self):
		print("[INFO] reading the video...")
		while True:
			(grabbed, frame) = self.vs.read()
			if not grabbed:
				break
			self.list_frames.append(frame)
		assert 0 < len(self.list_frames), "[WARNING] video empty, check the file name..."

	def detect_faces(self):
		print("[INFO] detecting faces...")
		for id_frame, frame in enumerate(self.list_frames):

			if self.W is None or self.H is None:
				(self.H, self.W) = frame.shape[:2]
			gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

			rects = self.detector(gray, 0)

			# for each face
			for (i, rect) in enumerate(rects):
				# convert dlib's rectangle to a OpenCV-style bounding box
				# [i.e., (x, y, w, h)], then draw the face bounding box
				(x, y, w, h) = face_utils.rect_to_bb(rect)
				# upsizing the rectangle
				modif_factor = 0.2
				modif_w = int(modif_factor * w)
				x -= modif_w
				w += 2*modif_w
				modif_h = int(modif_factor * h)
				y -= modif_h
				h += 2*modif_h

				self.list_of_faces.append(self.add_face_to_list(frame[y:y+h, x:x+w], id_frame, i, [x, y, w, h]))

	def load_classifier(self):
		# Load the model and its pretrained weights
		print("[INFO] loading the model...")
		self.classifier = MesoInception4()
		self.classifier.load(self.model_path)

	def gan_analysis(self):
		print("[INFO] analyzing the video...")
		for face in self.list_of_faces:

			frame = self.list_frames[face["id_frame"]]
			x, y, w, h = face["coord"]
			i = face["id_face"]

			pred = self.classifier.predict(np.expand_dims(face["image_resized"], axis=0)/255.)
			pred = float(pred)

			gan_detected = False
			if pred < self.is_gan_threshold:
				gan_detected = True

			# BGR instead of RGB because it is cv2
			color_is_gan = (0, 255, 0) if not gan_detected else (0, 0, 255)

			if gan_detected:
				frame[y:y+h, x:x+w] = get_heatmap_detection(self.classifier.model, face["image"])

			cv2.rectangle(frame, (x, y), (x + w, y + h), color_is_gan, 3)
			cv2.putText(frame, "Face #{}".format(i + 1), (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1., color_is_gan, 3)

	def save_video(self):
		print("[INFO] saving the video...")
		for frame in self.list_frames:
			# check if the video writer is None
			if self.writer is None:
				# initialize our video writer
				fourcc = cv2.VideoWriter_fourcc(*"MJPG")
				self.writer = cv2.VideoWriter(self.name_of_video[:-4] + "_analyzed.avi", fourcc, self.fps, (self.W, self.H), True)
			# write the output frame to disk
			self.writer.write(frame)

	def clean_end(self):
		# release the file pointers
		print("[INFO] cleaning up...")
		self.writer.release()
		self.vs.release()

	def main(self):
		self.load_face_detector()
		self.load_video()
		self.read_video()
		self.detect_faces()
		self.load_classifier()
		self.gan_analysis()
		self.save_video()
		self.clean_end()


if __name__ == "__main__":

	name_of_video = r"D:\deepfakes.gif"
	model_path = "models/MesoInception_DF.h5"

	analyzer = VideoGanAnalyzer(name_of_video, model_path)
	analyzer.main()

