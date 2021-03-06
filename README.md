Summary
=======
This github is studying for me about TensorFlow

Chapter 2
=========
1. MNIST
2. Softmax Regression

Chapter 3
=========
1. Computation Graphs
	* What is a Computation Graph?
	* The Benefits of Graph Computations
2. Graphs, Session, Fetches
	* Creating a Graph
	* Creating a Session and Running it
	* Constructing and Managing Our Graph
	* Fetches
3. Flowing Tensor
	* Nodes Are Operations, Edged Are Tensor Objects
	* Data Types
	* Tensor Arrays and Shapes
	* Names
4. Variables, Placeholders, Simple Optimization
	* Variables
	* Placeholders
	* Optimization

Chapter 4
=========
1. CNN
2. MNIST_TAKE2
	* Convolution
	* Pooling
	* Dropout
	* The Model
3. CIFAR10
	* Loading the CIFAR10 Dataset
	* Simple CIFAR10 Models

Chapter 5
=========
1. RNN
	* Vanilla RNN Implementation
	* TensorFlow Built-in RNN Functions
2. RNN for Text Sequences
	* Test Sequences
	* Supervised Word Embeddings
	* LSTM and Using Sequence Length
	* Training Embeddings and the LSTM Classifier

Chapter 6
=========
1. Word Embeddings
2. word2vec
	* skip-grams
	* Embeddings in TensorFlow
	* The Noise-Constrative Estimation(NCE) Loss Function
	* Learning Rate Decay
	* Training and Visualizing with TensorBoard
	* Checking Out Our Embeddings
3. Pretrained Embeddings, Advanced RNN
	* Pretrained Word Embeddings
	* Bidirectional RNN and GRU Cells

Chapter 7 (Holding)
===================
1. contrib.learn
	* Linear Regression
	* DNN Classfier
	* FeatureColumn
	* Homemade CNN with contrib.learn
2. TFLearn
	* Installation
	* CNN
	* RNN
	* Keras
	* Pretrained models with TF-Slim

Chapter 8
=========
1. TFRecords
	* Writing with TFRecordWriter
2. Queues
	* Enqueuing and Dequeuing
	* Multithreading
	* Coordinator and QueueRunner
3. A Full Multithreaded Input Pipeline
	* tf.train.string_input_producer() and tf.TFRecordReader()
	* tf.train.shuffle_batch()
	* tf.train.start_queue_runners() and Wrapping up

Chapter 9
=========
1. Distributed Computing
	* Where Does the Parallelization Take Place?
	* What is the Goal of Parallelization
2. TensorFlow Elements
	* tf.app.flags
	* Clusters and Servers
	* Replicating a Computational Graph Across Devices
	* Managed Sessions
	* Device Placement
3. Distributed Example
