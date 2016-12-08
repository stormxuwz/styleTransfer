import os
os.environ["CUDA_VISIBLE_DEVICES"]=""
import tensorflow as tf
import pickle
import scipy.misc
import numpy as np
# slim = tf.contrib.slim
from tensorflow.python.platform import gfile

# InceptionV4/InceptionV4/Conv2d_1a_3x3/Conv2D

def content(layer):
	# layer = tf.get_default_graph().get_tensor_by_name(layerName)
	return layer

def style(layer, size):
	# layer = tf.get_default_graph().get_tensor_by_name(layerName)
	_, height, width, number = map(lambda i: i.value, layer.get_shape())
	if size is None:
		size = height * width * number
	else:
		size = size[1]*size[2]*size[3]
	feats = tf.reshape(layer, (-1, number))
	gram = tf.matmul(tf.transpose(feats), feats)/size  # gram is a matrix
	return gram

def getFeatureTensors(layerNames, featureType = "content", size = None):
	# contentImage has been normalized
	features = {}
	for layerName in layerNames:
		# print "2",layerName
		layer = tf.get_default_graph().get_tensor_by_name(layerName)
		if featureType == "content":
			tensor = content(layer)
		elif featureType == "style":
			tensor = style(layer,size[layerName])
		# print tensor
		
		features[layerName] = tensor

	# print features
	
	return features


def stylize(contentImage, styleImage, contentLayerNames, styleLayerNames):
	tf.reset_default_graph()

	with tf.Session() as sess:
		with gfile.FastGFile("./plainModel/inception_v4.pb",'rb') as f:
			graph_def = tf.GraphDef()
			graph_def.ParseFromString(f.read())
			tf.import_graph_def(graph_def, name='')
		# tf.import_graph_def(graph_def, input_map={"InputImage:0": inputImage})

	  	endPoints = pickle.load(open("./plainModel/inception4_resave_endPoints.p"))
	  	print endPoints
		inputImage = tf.get_default_graph().get_tensor_by_name("InputImage:0")

		image_size = contentImage.shape
		# imageShape = (512,512,3)
		contentImage = scipy.misc.imresize(contentImage, image_size).astype(np.float)
		styleImage = scipy.misc.imresize(styleImage, image_size).astype(np.float)

		# print "available node:",endPoints.keys()
		print image_size
		# print tf.trainable_variables()
		# mean_pix = np.mean(contentImage,axis = (0,1))
		mean_pix = np.array([ 104.00698793,  116.66876762,  122.67891434])
		# print mean_pix

		contentImage = contentImage - mean_pix
		styleImage = styleImage - mean_pix
		contentImage = np.expand_dims(contentImage, axis = 0).astype(np.float)
		styleImage = np.expand_dims(styleImage, axis = 0).astype(np.float)

		# print inputImage
		contentTensor = getFeatureTensors([endPoints[k] for k in endPoints.keys() if k in contentLayerNames], "content")

		fullLayerNames = [endPoints[k] for k in endPoints.keys() if k in styleLayerNames]
		evalRes = sess.run(fullLayerNames, {inputImage:contentImage})
		shapes = [i.shape for i in evalRes]
		size = dict(zip(fullLayerNames,shapes))
		print size

		styleTensor = getFeatureTensors([endPoints[k] for k in endPoints.keys() if k in styleLayerNames], "style",size)
		
		
		# print endPoints.keys()
		print [endPoints[k] for k in endPoints.keys() if k in styleLayerNames]

		contentTarget = sess.run(contentTensor, {inputImage:contentImage})
		styleTarget = sess.run(styleTensor, {inputImage:styleImage})

	
	tf.reset_default_graph()

	with tf.Session() as sess:
		# image_size = 224
		# inputImage = tf.get_variable(name = "inputImage", shape = (1,image_size[0], image_size[1],3), \
					# initializer = tf.random_normal_initializer(mean=0.0, stddev=10, dtype=tf.float32))
		inputImage = tf.Variable(styleImage.astype(np.float32),name = "inputImage")
		sess.run(tf.initialize_variables([inputImage]))
		graph_def = tf.GraphDef()
		with gfile.FastGFile("./plainModel/inception_v4.pb",'rb') as f:
			graph_def.ParseFromString(f.read())
			# tf.import_graph_def(graph_def, name='')
			tf.import_graph_def(graph_def, input_map={"InputImage:0": inputImage})
		
		# writer = tf.train.SummaryWriter("log_tb",sess.graph)
		contentLoss = 0.0
		styleLoss = 0.0	
		contentTensor = getFeatureTensors(["import/"+endPoints[k] for k in endPoints.keys() if k in contentLayerNames], "content")

		size = dict(zip(["import/"+endPoints[k] for k in endPoints.keys() if k in styleLayerNames],shapes))

		styleTensor = getFeatureTensors(["import/"+endPoints[k] for k in endPoints.keys() if k in styleLayerNames], "style",size)
			
		for key in contentTarget.keys():
			contentLoss += tf.nn.l2_loss(contentTensor["import/"+key] - contentTarget[key])

		styleLayerWeights = 1.0/len(styleLayerNames)

		for key in styleTarget.keys():
			styleLoss += styleLayerWeights*tf.nn.l2_loss(styleTensor["import/"+key] - styleTarget[key])

		loss = contentLoss+1000.0*styleLoss
		print loss
		train_op = tf.train.AdamOptimizer(learning_rate=10).minimize(loss, var_list = [inputImage])
		sess.run(tf.initialize_all_variables())	
		print "created graph"
		for i in range(200):
			print i
			sess.run(train_op)
			
		newImage = sess.run(inputImage)
		newImage[0,:,:,:] +=mean_pix
		return newImage


if __name__ == '__main__':
	contentImage = scipy.misc.imread("./Examples/content_1.jpg").astype(np.float)
	styleImage = scipy.misc.imread("./Examples/style_1.jpg").astype(np.float)
	
	contentLayerNames = ['Mixed_5a']
	styleLayerNames = ['Conv2d_1a_3x3','Conv2d_2a_3x3','Conv2d_2b_3x3']


	resultImage = stylize(contentImage, styleImage, contentLayerNames, styleLayerNames)
	resultImage = resultImage[0,:,:,:]
	img = np.clip(resultImage, 0, 255).astype(np.uint8)
	scipy.misc.imsave("./test_results.jpg", img)