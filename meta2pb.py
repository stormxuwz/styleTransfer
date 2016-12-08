import os
os.environ["CUDA_VISIBLE_DEVICES"]=""
import numpy as np
import tensorflow as tf
import urllib2
import pickle
# from datasets import imagenet
from nets import inception,vgg, resnet_v1
from tensorflow.python.platform import gfile
from tensorflow.python.framework import graph_util
# from preprocessing import inception_preprocessing
import scipy.misc

slim = tf.contrib.slim

def chooseModel(model):
	if model == "vgg_19":
		modelPath = "./models/vgg_19.ckpt"
		image_size = vgg.vgg_19.default_image_size
		varScope = vgg.vgg_arg_scope()
		modelGraph = vgg.vgg_19
		scopeName = "vgg_19"

	elif model == "vgg_16":
		modelPath = "./models/vgg_16.ckpt"
		image_size = vgg.vgg_16.default_image_size
		varScope = vgg.vgg_arg_scope()
		modelGraph = vgg.vgg_16
		scopeName = "vgg_16"

	elif model == "inception4":
		modelPath = "./models/inception_v4.ckpt"
		image_size = inception.inception_v4.default_image_size
		varScope = inception.inception_v4_arg_scope()
		modelGraph = inception.inception_v4
		scopeName = "InceptionV4"

	elif model == "resnet_v1_152":
		modelPath = "./models/resnet_v1_152.ckpt"
		image_size = resnet_v1.resnet_v1.default_image_size
		varScope = resnet_v1.resnet_arg_scope()
		modelGraph = resnet_v1.resnet_v1_152
		scopeName = "resnet_v1_152"

	elif model == "inception_resnet_v2":
		modelPath = "./models/inception_resnet_v2_2016_08_30.ckpt"
		image_size = inception.inception_resnet_v2.default_image_size
		varScope = inception.inception_resnet_v2_arg_scope()
		modelGraph = inception.inception_resnet_v2
		scopeName = "InceptionResnetV2"

	else:
		raise ValueError("not implemented")
	return modelPath, image_size, varScope, modelGraph, scopeName

def createGraph(model, initialization):
	modelPath, image_size, varScope, modelGraph, scopeName = chooseModel(model)
	inputs = tf.get_variable(name = "inputImage", shape = (1,image_size,image_size,3), \
				initializer = tf.random_normal_initializer(mean=0.0, stddev=0.1, dtype=tf.float32))
	with slim.arg_scope(varScope):
			logits, end_points = modelGraph(inputs,is_training=False)
	# print scopeName
	init_fn = slim.assign_from_checkpoint_fn(modelPath,slim.get_model_variables(scopeName))

	print "graph created"
	return inputs, end_points, init_fn,image_size


def writeGraph(model):
	# function to transfer data from meta file to pb file
	tf.reset_default_graph()
	inputs, end_points, init_fn, image_size = createGraph(model, None)

	with tf.Session() as sess:
		init_fn(sess)

		sess.run(tf.initialize_variables([inputs]))

		if "vgg" in model:
			endNodeName = "%s/fc8" %(model,)
		elif model == "resnet_v1_152":
			endNodeName = "predictions"
		else:
			endNodeName = "Predictions"

		output_graph_def = graph_util.convert_variables_to_constants(sess, sess.graph_def, \
			[end_points[endNodeName].name.split(":")[0]])

	with tf.gfile.GFile("./plainModel/%s_model.pb" %(model,), "wb") as f:
		f.write(output_graph_def.SerializeToString())
		print("%d ops in the final graph." % len(output_graph_def.node))


if __name__ == '__main__':
	modelList = ["vgg_19","vgg_16","inception4","resnet_v1_152","inception_resnet_v2"]
	# save the model from meta graph to pb file
	for model in modelList:
		print model
		tf.reset_default_graph()
		writeGraph(model)
