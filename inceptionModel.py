# using 
import os
os.environ["CUDA_VISIBLE_DEVICES"]=""  # comment out to use GPU
import tensorflow as tf
import pickle
import scipy.misc
import numpy as np
# slim = tf.contrib.slim
# modelList = ["vgg_19","vgg_16","inception4","resnet_v1_152","inception_resnet_v2"]
from tensorflow.python.platform import gfile

def content(layer):
	return layer

def style(layer, size):
	# return layer
	print map(lambda i: i.value, layer.get_shape())
	_, height, width, number = map(lambda i: i.value, layer.get_shape())
	# if size is None:
	size2 = height * width * number
	# else:
		# size = size[1]*size[2]*size[3]
	feats = tf.reshape(layer, (-1, number))
	gram = tf.matmul(tf.transpose(feats), feats)/size2  # gram is a matrix
	print gram
	return gram

def getFeatureTensors(layerNames, featureType = "content", size = None):
	# contentImage has been normalized
	features = {}
	for layerName in layerNames:
		layer = tf.get_default_graph().get_tensor_by_name(layerName)
		if featureType == "content":
			tensor = content(layer)
		elif featureType == "style":
			if size is not None:
				tensor = style(layer,size[layerName])
			else:
				tensor = style(layer, None)
		features[layerName] = tensor
	return features


def stylize(contentImage, styleImage, contentLayerNames, styleLayerNames,learning_rate):
	image_size = (299,299,3)
	# inputImage = tf.get_default_graph().get_tensor_by_name("ResizeBilinear:0")

	# preprocessing images
	contentImage = scipy.misc.imresize(contentImage, image_size).astype(np.float32)
	styleImage = scipy.misc.imresize(styleImage, image_size).astype(np.float32)
	
	scipy.misc.imsave("./contentImage.jpg", contentImage)
	scipy.misc.imsave("./styleImage.jpg", styleImage)


	# mean_pix = np.mean(contentImage,axis = (0,1))
	# std_pix = np.std(contentImage, axis = (0,1))
	# mean_pix = 0
	# mean_pix = np.mean(contentImage,axis = (0,1))
	mean_pix = np.array([ 104.00698793,  116.66876762,  122.67891434])
	std_pix = 1

	contentImage = (contentImage - mean_pix)/std_pix
	styleImage = (styleImage - mean_pix)/std_pix


	contentImage = np.expand_dims(contentImage.astype(np.float32), axis = 0)
	styleImage = np.expand_dims(styleImage.astype(np.float32), axis = 0)


	tf.reset_default_graph()
	inputImage = tf.Variable(styleImage, name="W",dtype = tf.float32)

	graph_def = tf.GraphDef()
	with gfile.FastGFile("./plainModel/tensorflow_inception_graph.pb",'rb') as f:
			graph_def.ParseFromString(f.read())
			tf.import_graph_def(graph_def, input_map={"Mul:0": inputImage})
	

	# endPoints ={
	#   	"mixed_10":"mixed_10/join:0",
	#   	"mixed_8":"mixed_8/join:0",
	#   	"mixed_7":"mixed_7/join:0",
	#   	"mixed_6":"mixed_6/join:0",
	#   	"mixed_5":"mixed_5/join:0",
	#   	"mixed_4":"mixed_4/join:0",
	#   	"mixed_3":"mixed_3/join:0",
	#   	"mixed_2":"mixed_2/join:0",
	#   	"mixed_1":"mixed_1/join:0",
	#   	"mixed":"mixed/join:0"}
	# print [n.name for n in tf.get_default_graph().as_graph_def().node]
 	
	contentTensor = getFeatureTensors(["import/"+k for k in contentLayerNames], "content")
	
	# print "mean_pix": mean_pix
	# print "contentImage:",contentImage
	# print "styleImage:", styleImage


	#evalRes = sess.run(fullLayerNames, {inputImage:contentImage})
	#shapes = [i.shape for i in evalRes]
	#size = dict(zip(fullLayerNames,shapes))
	size = None
	styleTensor = getFeatureTensors(["import/"+k for k in styleLayerNames], "style",size)
	
	# print endPoints.keys()

	with tf.Session() as sess:
		writer = tf.train.SummaryWriter("log_tb",sess.graph)
		contentTarget = sess.run(contentTensor, {inputImage:contentImage})
		styleTarget = sess.run(styleTensor, {inputImage:styleImage})
		# print "hello",sess.run("import/Mul/y:0")
	# construct loss
	# print contentTarget
	
	tf.reset_default_graph()

	graph_def = tf.GraphDef()
	inputImage = tf.get_variable(name = "inputImage", shape = (1,image_size[0], image_size[1],3), initializer = tf.random_normal_initializer(mean=0.0, stddev=100, dtype=tf.float32))
	# inputImage = tf.Variable(contentImage, name="W")
	with gfile.FastGFile("./plainModel/tensorflow_inception_graph.pb",'rb') as f:
			graph_def.ParseFromString(f.read())
			tf.import_graph_def(graph_def, input_map={"Mul:0": inputImage})

	contentLoss = 0.0
	styleLoss = 0.0	
	contentTensor = getFeatureTensors(["import/"+k for k in contentLayerNames], "content")

	# size = dict(zip(["import/"+endPoints[k] for k in endPoints.keys() if k in styleLayerNames],shapes))
	size = None
	styleTensor = getFeatureTensors(["import/"+k for k in styleLayerNames], "style",size)
	
	contentLayerWeights = 1.0/len(contentLayerNames)
	print "contentLayers:",contentTarget.keys()

	for key in contentTarget.keys():
		contentLoss += contentLayerWeights*tf.nn.l2_loss(contentTensor[key] - contentTarget[key])
		# print contentTensor["import/"+key], contentTarget[key].shape
	styleLayerWeights = 1.0/len(styleLayerNames)
	
	print "styleLayers:",styleTarget.keys()
	
	for key in styleTarget.keys():
		styleLoss += styleLayerWeights*tf.nn.l2_loss(styleTensor[key] - styleTarget[key])
		# print styleTensor["import/"+key], styleTarget[key].shape
	styleLoss = 1e3*styleLoss
	
	loss = styleLoss + contentLoss # styleLoss

	# tv_y_size = image_size[0]
	# tv_x_size = image_size[1]
	# tv_loss = 0.1 * 2 * ((tf.nn.l2_loss(inputImage[:,1:,:,:] - inputImage[:,:image_size[0]-1,:,:]) /tv_y_size) +(tf.nn.l2_loss(inputImage[:,:,1:,:] - inputImage[:,:,:image_size[1]-1,:]) /tv_x_size))


	train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

	with tf.Session() as sess:
		sess.run(tf.initialize_all_variables())	
		print "created graph"
		
		print sess.run([loss,contentLoss, styleLoss])

		for i in range(601):
			sess.run(train_op)
			# print sess.run([loss,styleLoss*500, contentLoss])
			
			if i%100 == 0:
				print i
				print sess.run([loss,contentLoss, styleLoss])
				newImage = sess.run(inputImage)
				newImage[0,:,:,:] = newImage[0,:,:,:]*std_pix
				newImage[0,:,:,:] +=mean_pix
				scipy.misc.imsave("./results/inceptionM_results_test_%d.jpg" %(i,), np.clip(newImage[0,:,:,:], 0, 255).astype(np.uint8))
			# print sess.run(contentTensor)
		newImage = sess.run(inputImage)
		newImage[0,:,:,:] = newImage[0,:,:,:]*std_pix
		newImage[0,:,:,:] +=mean_pix
		return newImage

if __name__ == '__main__':
	import json,sys
	learning_rate = sys.argv[1]
	with open('./config/inception_config.json') as data_file:    
		layerConfig = json.load(data_file)

	if True:
		k = "config_1"
	# for k in layerConfig.keys():
		v = layerConfig[k]
		print k
		contentLayerNames = v["contentLayerNames"] #['mixed_3/join:0']
		styleLayerNames = v["styleLayerNames"] #['mixed_1/join:0','mixed_1/join:0','mixed_1/join:0','mixed_1/join:0']

		for i in range(3,4):
			for j in range(1,2):
				contentImage = scipy.misc.imread("./Examples/content_%d.jpg" %(i,)).astype(np.float32)
				styleImage = scipy.misc.imread("./Examples/style_%d.jpg" %(j,)).astype(np.float32)
			
				resultImage = stylize(contentImage, styleImage, contentLayerNames, styleLayerNames,float(learning_rate))
				resultImage = resultImage[0,:,:,:]
				img = np.clip(resultImage, 0, 255).astype(np.uint8)
				scipy.misc.imsave("./results/inception_results_%s_content_%d_style_%d_rl_%s.jpg" %(k,i,j,learning_rate), img)