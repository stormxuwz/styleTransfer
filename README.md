# neural-style
We write the style transfer code based on https://github.com/anishathalye/neural-style

(1) download the tensorflow VGG19 model from  http://www.vlfeat.org/matconvnet/models/beta16/imagenet-vgg-verydeep-19.mat
(2) Other models can be downloaded from 
(2) Put the model into ./models/ folder
(3) modify the myStylize.py to incorporate the content image and style image
(4) Run the myStylize.py

For using models from pretrained models including inception, res-net, vgg-19, vgg-16 from tensorflow/slim, download the pb files from https://drive.google.com/open?id=0B-YQJ1l195yzMUItcFBvaTV5aFk and run generalizedStylizer.py

note1: The pb files are generated by meta2pb.py, based the meta graph model downloaded from https://github.com/tensorflow/models/tree/master/slim. (The meta graph requires newest version of tensorflow, so I changed the meta graph to pb, hope to have some compatibility for older tensor, but it seems not working well with 0.8 version on the cluster. It seems pb files may still not back-compatible if using some high-level API when created)

note2: tensorflow_inception_graph.pb is the V3 version of inception used as examples for tensorflow, which are compatible with older tensorflow

note3: inception_v4.pb don't have restrictions on the input size, while other pb file models have

## Requirements

* [TensorFlow](https://www.tensorflow.org/versions/master/get_started/os_setup.html#download-and-setup)
* [NumPy](https://github.com/numpy/numpy/blob/master/INSTALL.rst.txt)
* [SciPy](https://github.com/scipy/scipy/blob/master/INSTALL.rst.txt)
* [Pillow](http://pillow.readthedocs.io/en/3.3.x/installation.html#installation)
* [Pre-trained VGG network][net] (MD5 `8ee3263992981a1d26e73b3ca028a123`)
