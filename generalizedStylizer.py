'''
Script to utlized the saved Pb file to do style transfer
'''

import os
os.environ["CUDA_VISIBLE_DEVICES"]=""
import tensorflow as tf
import pickle
import scipy.misc
import numpy as np
# slim = tf.contrib.slim
modelList = ["vgg_19","vgg_16","inception4","resnet_v1_152","inception_resnet_v2"] # available models

from tensorflow.python.platform import gfile

def content(layer):
	# layer = tf.get_default_graph().get_tensor_by_name(layerName)
	return layer

def style(layer):
	# layer = tf.get_default_graph().get_tensor_by_name(layerName)
	_, height, width, number = map(lambda i: i.value, layer.get_shape())
	size = height * width * number
	feats = tf.reshape(layer, (-1, number))
	gram = tf.matmul(tf.transpose(feats), feats)/size  # gram is a matrix
	return gram

def getFeatureTensors(layerNames, featureType = "content"):
	# contentImage has been normalized
	features = {}
	for layerName in layerNames:
		layer = tf.get_default_graph().get_tensor_by_name(layerName)
		if featureType == "content":
			tensor = content(layer)
		elif featureType == "style":
			tensor = style(layer)
		features[layerName] = tensor
	return features

def stylize(model, contentImage, styleImage, contentLayerNames, styleLayerNames):
	tf.reset_default_graph()

	with tf.Session() as sess:
		# saver = tf.train.import_meta_graph('./plainModel/%s_resave.meta' %(model,))
		graph_def = tf.GraphDef()
		with gfile.FastGFile("./plainModel/%s_model.pb" %(model,),'rb') as f:
			graph_def.ParseFromString(f.read())
			tf.import_graph_def(graph_def, name='')

	  	endPoints = pickle.load(open("./plainModel/%s_resave_endPoints.p" %(model,)))
	  	
		inputImage = tf.get_default_graph().get_tensor_by_name("inputImage:0")
		image_size = inputImage.get_shape()[2] # get the model style image

		contentImage = scipy.misc.imresize(contentImage, (image_size,image_size,3))
		styleImage = scipy.misc.imresize(styleImage, (image_size,image_size,3))

		# print "available node:",endPoints.keys()
		print image_size
		mean_pix = np.array([ 104.00698793,  116.66876762,  122.67891434]) # mean from inceptionv3
		std_pix = 1 #doing scale

		contentImage = contentImage - mean_pix
		styleImage = styleImage - mean_pix

		contentImage = np.expand_dims(contentImage,axis = 0).astype(np.float32)
		styleImage = np.expand_dims(styleImage,axis = 0).astype(np.float32)

		contentTensor = getFeatureTensors([endPoints[k] for k in endPoints.keys() if k in contentLayerNames], "content")
		styleTensor = getFeatureTensors([endPoints[k] for k in endPoints.keys() if k in styleLayerNames], "style")
		
		contentTarget = sess.run(contentTensor, {inputImage:contentImage})
		styleTarget = sess.run(styleTensor, {inputImage:styleImage})

	tf.reset_default_graph()

	with tf.Session() as sess:
		# image_size = 299

		# set random initial image
		# inputImage2 = tf.get_variable(name = "inputImage2", shape = (1,image_size,image_size,3), initializer = tf.random_normal_initializer(mean=0.0, stddev=10, dtype=tf.float32))

		inputImage2 = tf.Variable(contentImage,name = "inputImage2") # change to style image to initialize from style image
		graph_def = tf.GraphDef()
		with gfile.FastGFile("./plainModel/%s_model.pb" %(model,),'rb') as f:
			graph_def.ParseFromString(f.read())
			_ = tf.import_graph_def(graph_def, input_map={"inputImage:0": inputImage2}) # replace the input nodes with variable
	
		contentLoss = 0.0
		styleLoss = 0.0	
		contentTensor = getFeatureTensors(["import/"+endPoints[k] for k in endPoints.keys() if k in contentLayerNames], "content")
		styleTensor = getFeatureTensors(["import/"+endPoints[k] for k in endPoints.keys() if k in styleLayerNames], "style")
			
		for key in contentTarget.keys():
			contentLoss += tf.nn.l2_loss(contentTensor["import/"+key] - contentTarget[key])

		styleLayerWeights = 1.0/len(styleLayerNames)

		for key in styleTarget.keys():
			styleLoss += styleLayerWeights*tf.nn.l2_loss(styleTensor["import/"+key] - styleTarget[key])

		loss = contentLoss+1000.0*styleLoss

		temp = set(tf.global_variables())

		# print tf.GraphKeys.GLOBAL_VARIABLES
		train_op = tf.train.AdamOptimizer(learning_rate=1e1).minimize(loss)
		sess.run(tf.initialize_variables(set(tf.all_variables())-temp))
		sess.run(tf.initialize_variables([inputImage2]))

		print sess.run(inputImage2).sum()
		print "created graph"
		history = []
		lossHistory = []
		cl,sl= sess.run([contentLoss,1000.0*styleLoss])
		print cl,sl
		for i in range(1000):
			print i
			sess.run(train_op)
			
			if i % 100 == 0:
				newImage, cl,sl= sess.run([inputImage2,contentLoss,1000.0*styleLoss])
				print newImage.sum(),cl,sl
				newImage[0,:,:,:] = newImage[0,:,:,:]*std_pix
				newImage[0,:,:,:] +=mean_pix
				scipy.misc.imsave("./results/inception_results_test_%d.jpg" %(i,), np.clip(newImage[0,:,:,:], 0, 255).astype(np.uint8))

		newImage = sess.run(inputImage2)
		return newImage, history,lossHistory


if __name__ == '__main__':
	contentImage = scipy.misc.imread("./Examples/content_1.jpg").astype(np.float32)
	styleImage = scipy.misc.imread("./Examples/style_1.jpg").astype(np.float32)
	
	# contentLayerNames = ['vgg_19/conv4/conv4_2']
	# styleLayerNames = ['vgg_19/conv4/conv4_1', 'vgg_19/conv5/conv5_1','vgg_19/conv1/conv1_1', 'vgg_19/conv2/conv2_1', 'vgg_19/conv3/conv3_1']
	
	contentLayerNames = ['Conv2d_1a_3x3']
	styleLayerNames = ['Conv2d_2b_3x3',"Conv2d_1a_3x3","Mixed_3a","Mixed_4a"]

	resultImage,history,lossHistory = stylize("inception4", contentImage, styleImage, contentLayerNames, styleLayerNames)
	resultImage = resultImage[0,:,:,:]

	for i,image in enumerate(history):
		print image.shape
		img = np.clip(image, 0, 255).astype(np.uint8)
		scipy.misc.imsave("./test_results_%d.jpg" %(i,), img)
	
	print lossHistory
