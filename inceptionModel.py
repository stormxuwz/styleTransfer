import os
os.environ["CUDA_VISIBLE_DEVICES"]="" #comment this line to use GPU
import tensorflow as tf
import pickle
import scipy.misc
import numpy as np
from tensorflow.python.platform import gfile


def content(layer):
	# layer is a tensor represent the feature map
	return layer

def style(layer, size):
	# layer is a tensor represent the feature map
	# size is the size of each feature map
	_, height, width, number = map(lambda i: i.value, layer.get_shape())
	if size is None:
		size = height * width * number
	else:
		size = size[1]*size[2]*size[3]
	feats = tf.reshape(layer, (-1, number))
	gram = tf.matmul(tf.transpose(feats), feats)/size  # gram is a matrix
	return gram

def getFeatureTensors(layerNames, featureType = "content", size = None):
	# layerNames: layer names
	# featureType: get content representation or style representation
	# size of each layer, necessary if the graph can't infer by itself
	features = {}
	for layerName in layerNames:
		layer = tf.get_default_graph().get_tensor_by_name(layerName)
		if featureType == "content":
			tensor = content(layer)
		elif featureType == "style":
			tensor = style(layer,size[layerName])
		features[layerName] = tensor
	
	return features


def stylize(model,contentImage, styleImage, contentLayerNames, styleLayerNames, image_size = None, iniMethod = "random"):
	# model: choose from "InceptionV1","InceptionV3","InceptionV4"
	# contentImage, styleImage: images
	# contentLayerNames: which layers are used to construct the content representation
	# styleLayerNames: which layers are used to construct the style representation

	tf.reset_default_graph()
	with gfile.FastGFile("./plainModel/%s.pb" %(model,),'rb') as f:
			graph_def = tf.GraphDef()
			graph_def.ParseFromString(f.read())
			tf.import_graph_def(graph_def, name='')

	with tf.Session() as sess:
		if model == "InceptionV3":
			inputImage = tf.get_default_graph().get_tensor_by_name("ResizeBilinear:0")
		else:
			inputImage = tf.get_default_graph().get_tensor_by_name("InputImage:0")
		if image_size is None:
			image_size = contentImage.shape
		# image_size = (224,224,3)
		contentImage = scipy.misc.imresize(contentImage, image_size).astype(np.float)
		styleImage = scipy.misc.imresize(styleImage, image_size).astype(np.float)

		# mean_pix = np.mean(contentImage,axis = (0,1)) # normalize by the mean values
		mean_pix = np.array([ 104.00698793,  116.66876762,  122.67891434]) # normalized from XXX
		# print mean_pix

		contentImage = contentImage - mean_pix
		styleImage = styleImage - mean_pix
		contentImage = np.expand_dims(contentImage, axis = 0).astype(np.float)
		styleImage = np.expand_dims(styleImage, axis = 0).astype(np.float)

		# for variable in tf.all_variables():
			# print variable
		# print [n.name for n in tf.get_default_graph().as_graph_def().node]. # print all the tensors

		# print inputImage
		contentTensor = getFeatureTensors(contentLayerNames, "content")

		 # calculate the size of style representations
		fullLayerNames = styleLayerNames
		evalRes = sess.run(fullLayerNames, {inputImage:contentImage})
		shapes = [i.shape for i in evalRes]
		size = dict(zip(fullLayerNames,shapes))
		print size

		styleTensor = getFeatureTensors(styleLayerNames,"style",size)
		
		contentTarget = sess.run(contentTensor, {inputImage:contentImage})
		styleTarget = sess.run(styleTensor, {inputImage:styleImage})

	# construct loss
	# print contentTarget
	tf.reset_default_graph()
	graph_def = tf.GraphDef()
	
	if iniMethod == "random":
		inputImage = tf.get_variable(name = "inputImage", shape = (1,image_size[0], image_size[1],3), \
					initializer = tf.random_normal_initializer(mean=0.0, stddev=10, dtype=tf.float32))
	elif iniMethod == "content":
		inputImage = tf.Variable(contentImage.astype(np.float32),name = "inputImage")
	elif iniMethod == "style":
		inputImage = tf.Variable(styleImage.astype(np.float32),name = "inputImage")

	with gfile.FastGFile("./plainModel/%s.pb" %(model,),'rb') as f:
		graph_def.ParseFromString(f.read())
			# tf.import_graph_def(graph_def, name='')
		# replace the InputImage part by a trainable variable
		if model == "InceptionV3":
			tf.import_graph_def(graph_def, input_map={"ResizeBilinear:0": inputImage})
		else:
			tf.import_graph_def(graph_def,input_map={"InputImage:0": inputImage}) 

	with tf.Session() as sess:
		# image_size = (224,224)
		sess.run(tf.initialize_variables([inputImage]))
		
		writer = tf.train.SummaryWriter("log_tb",sess.graph)
		contentLoss = 0.0
		styleLoss = 0.0	
		contentTensor = getFeatureTensors(["import/"+k for k in contentLayerNames], "content")

		size = dict(zip(["import/"+k for k in styleLayerNames],shapes))

		styleTensor = getFeatureTensors(["import/"+k for k in styleLayerNames], "style",size)

		for key in contentTarget.keys():
			contentLoss += 0.5*tf.nn.l2_loss(contentTensor["import/"+key] - contentTarget[key])

		styleLayerWeights = 1.0/len(styleLayerNames)

		for key in styleTarget.keys():
			styleLoss += 0.25*styleLayerWeights*tf.nn.l2_loss(styleTensor["import/"+key] - styleTarget[key])

		loss = contentLoss+1000.0*styleLoss
		# print loss
		train_op = tf.train.AdamOptimizer(learning_rate=10).minimize(loss, var_list = [inputImage])
		sess.run(tf.initialize_all_variables())	
		print "created graph"
		for i in range(1000):
			print i
			sess.run(train_op)
			
		newImage = sess.run(inputImage)
		newImage[0,:,:,:] +=mean_pix
		return newImage


if __name__ == '__main__':

	layerDict = {

	# InceptionV1 model have no restriction on the image size of 224
	# downloaded from https://github.com/beniz/deepdetect/issues/89
	"InceptionV1":{\
	"content":['InceptionV1/InceptionV1/MaxPool_3a_3x3/MaxPool:0'],
	"style":['InceptionV1/InceptionV1/Conv2d_1a_7x7/Relu:0',\
	'InceptionV1/InceptionV1/Conv2d_2c_3x3/Relu:0',\
	'InceptionV1/InceptionV1/MaxPool_3a_3x3/MaxPool:0',\
	"InceptionV1/InceptionV1/MaxPool_4a_3x3/MaxPool:0",\
	"InceptionV1/InceptionV1/MaxPool_5a_2x2/MaxPool:0"],
	"size":[224,224,3]},

	# InceptionV4 model have no restriction on the image size
	# Downloaded from https://github.com/beniz/deepdetect/issues/89
	"InceptionV4":{\
	"content":['InceptionV4/InceptionV4/Mixed_5a/concat:0'],
	"style":['InceptionV4/InceptionV4/Conv2d_1a_3x3/Relu:0',\
	'InceptionV4/InceptionV4/Conv2d_2a_3x3/Relu:0',\
	'InceptionV4/InceptionV4/Conv2d_2b_3x3/Relu:0'],
	"size":[299,299,3]},

	# InceptionV3 model is from http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz
	# InceptionV3 model have restrictions on the image size of 299
	"InceptionV3":{
	"content":["mixed_4/join:0"],
	"style":["conv:0","pool:0","conv_4:0","pool_1:0"],
	"size":[299,299,3]}
	}

	for model in ["InceptionV1","InceptionV4","InceptionV3"]:
		for i in range(1,2):	
			for j in range(1,2):
				for iniMethod in ["content","random","style"]:
					contentImage = scipy.misc.imread("./Examples/content_%d.jpg" %(i,)).astype(np.float)
					styleImage = scipy.misc.imread("./Examples/style_%d.jpg" %(j,)).astype(np.float)
		
					contentLayerNames = layerDict[model]["content"]
					styleLayerNames = layerDict[model]["style"]
					imageSize = layerDict[model]["size"]

					resultImage = stylize(model,contentImage, styleImage, contentLayerNames, styleLayerNames, imageSize,iniMethod)
					resultImage = resultImage[0,:,:,:]
					img = np.clip(resultImage, 0, 255).astype(np.uint8)
					scipy.misc.imsave("./%s_c_%d_s_%d_ini_%s.jpg" %(model,i,j,iniMethod), img)