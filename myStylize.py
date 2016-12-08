import os
os.environ["CUDA_VISIBLE_DEVICES"]=""
import tensorflow as tf
import numpy as np
import scipy.misc
import vgg
import time

initializer = tf.random_normal_initializer(mean=0.0, stddev=10, dtype=tf.float32)


config_CPU = tf.ConfigProto(device_count = {'CPU': 1,"GPU": 0})


def content(layer):
	'''
	function to construct content tensors
	'''
	return layer


def styles(layer, type = ""):
	'''
	function to construct style tensors
	'''
	_, height, width, number = map(lambda i: i.value, layer.get_shape())
	size = height * width * number
	feats = tf.reshape(layer, (-1, number))
	gram = tf.matmul(tf.transpose(feats), feats)/size  # gram is a matrix
	return gram
	# layerFilter = tf.transpose(layer,(2,1,0,3)) #[height, width, in_channel, output_channel]
	# layerFilter = layer[0,:,:,:]
	# return tf.nn.conv2d(layer,layerFilter, strides=[1, 1, 1, 1], padding='SAME')



def evalFeatureLayers(graph, tensor, myImage):
	# myImage is already preprocessed
	with tf.Session() as sess:
		res = sess.run(tensor,feed_dict={"inputImage:0": myImage})
		return res

def getFeatures(graph, layerNames = ["conv1_1"], featureType = "content", returnTensor = True, inputImage = None):
	# contentImage has been normalized
	features = {}

	for layerName in layerNames:
		# print "2",layerName
		layer = tf.get_default_graph().get_tensor_by_name(layerName+":0")
		if featureType == "content":
			tensor = content(layer)
		elif featureType == "style":
			tensor = styles(layer)
		# print tensor
		if returnTensor:
			features[layerName] = tensor
		else:
			layerValues = evalFeatureLayers(graph, tensor, inputImage)
			features[layerName] = layerValues

	return features


# get the content and styple representation

def stylize(contentImage, styleImage, contentLayerNames = ['conv4_2'], styleLayerNames = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']):

	# imageShape = contentImage.shape
	imageShape = (224,224,3)
	contentImage = scipy.misc.imresize(contentImage, imageShape)
	styleImage = scipy.misc.imresize(styleImage, imageShape)

	contentImage = np.expand_dims(contentImage, axis = 0)
	styleImage = np.expand_dims(styleImage, axis = 0)
	print imageShape
	# get the graph
	g = tf.Graph()
	with g.as_default():
		image = tf.placeholder('float', shape=(None,imageShape[0],imageShape[1],imageShape[2]),name = "inputImage")
		net, mean_pixel = vgg.net("./models/imagenet-vgg-verydeep-19.mat", image) # input as placeholder
		print mean_pixel
		contentImage = contentImage- mean_pixel
		styleImage = styleImage - mean_pixel

		contentTensor = getFeatures(g, layerNames = contentLayerNames, featureType = "content", returnTensor=True) # create new content tensor
		styleTensor = getFeatures(g, layerNames = styleLayerNames, featureType = "style", returnTensor=True)	# create new style tensor

		with tf.Session() as sess:
			contentFeatures = sess.run(contentTensor, feed_dict={"inputImage:0": contentImage}) # evaluate content tensor
			styleFeatures = sess.run(styleTensor, feed_dict={"inputImage:0": styleImage})	# evaluate style tensor
			
			print "'relu1_1'",sess.run('relu1_1:0',feed_dict ={"inputImage:0":contentImage}).sum()

		print "contentImage,",np.sum(contentImage)
		print [{k:np.sum(styleFeatures[k])} for k in styleFeatures.keys()]
	

	tf.reset_default_graph()
	print "training graph"
	# get a new graph
	g = tf.Graph()
	with g.as_default():
		writer = tf.train.SummaryWriter("log_tb",sess.graph)
		# image = tf.get_variable("initialImage", shape = (1,)+imageShape,initializer = initializer,dtype = tf.float32)
		image = tf.Variable(contentImage.astype(np.float32),name = "initialImage")
		# reconstruct the tensorflow graph with input as variable
		net, _ = vgg.net("./models/imagenet-vgg-verydeep-19.mat", image)

		contentLoss = 0.0
		styleLoss = 0.0

		for layerName in contentLayerNames:
			contentTensor = getFeatures(g, layerNames = [layerName], featureType = "content", returnTensor=True)
			contentLoss = contentLoss + tf.nn.l2_loss(contentFeatures[layerName] - contentTensor[layerName])
			
		styleLayerWeights = 1.0/len(styleLayerNames)
		for layerName in styleLayerNames:
			styleTensor = getFeatures(g, layerNames = [layerName], featureType = "style", returnTensor=True)
			styleLoss = styleLoss + styleLayerWeights*tf.nn.l2_loss(styleFeatures[layerName] - styleTensor[layerName])


		if contentLossExist:
			loss = contentLoss + 1000.0*styleLoss
		else:
			print "no content loss"
			loss = styleLoss

		train = tf.train.AdamOptimizer(1e1).minimize(loss)
		historyImage = []
		historyLoss = []
		with tf.Session() as sess:
			sess.run(tf.initialize_all_variables())
			start = time.time()

			resultImage = sess.run("initialImage:0")
			print resultImage.sum()
			resultImage[0,:,:,:]+= mean_pixel
			historyImage.append(resultImage)
			historyLoss.append(sess.run([loss,contentLoss,styleLoss]))

			for i in range(2000):
				print i
				sess.run(train)
				if i % 200 ==0:
					newImage, l= sess.run(["initialImage:0",loss])
					print newImage.sum(),l
					newImage[0,:,:,:] = newImage[0,:,:,:]
					newImage[0,:,:,:] +=mean_pixel
					scipy.misc.imsave("./results/vgg2_results_test_%d.jpg" %(i,), np.clip(newImage[0,:,:,:], 0, 255).astype(np.uint8))

			print time.time() -start
			# resultImage = sess.run("initialImage:0")
			# resultImage[0,:,:,:] += mean_pixel
			return historyImage, historyLoss

if __name__ == '__main__':
	import scipy.misc
	import matplotlib.pyplot as plt
	import sys
	import pickle

	contentLossExist = True
	# styleImage = sys.argv[2]
	# resultsIndex = sys.argv[3]

	contentLayerNames = ['relu4_2']
	styleLayerNames = ['relu4_1', 'relu5_1','relu1_1', 'relu2_1', 'relu3_1']

	for i in range(1,2):
		for j in range(1,2):
			contentImage = scipy.misc.imread("./Examples/content_%d.jpg" %(i,)).astype(np.float32)
			styleImage = scipy.misc.imread("./Examples/style_%d.jpg" %(j,)).astype(np.float32)
				
			historyImage, historyLoss = stylize(contentImage, styleImage, contentLayerNames = contentLayerNames, styleLayerNames = styleLayerNames)
			for k,img in enumerate(historyImage):
				scipy.misc.imsave("./results/vgg_noCL_content_%d_style_%d_seq_%d.jpg" %(i,j,k), img[0,:,:,:])

			pickle.dump(historyLoss, open("./results/lossHistory_noCL_content_%d_style_%d.p" %(i,j),"wb"))
