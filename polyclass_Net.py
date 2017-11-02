import tensorflow as tf 
from tensorflow.examples.tutorials.mnist import input_data # test data created by Dan

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True) # reads the training data set from the specified path

data_points = 1000 # amount to be loaded at one time
n_nodes_hl1 = 800 # nodes for layer 1
n_nodes_hl2 = 500 # nodes for layer 2
n_nodes_hl3 = 400 # nodes for layer 3
data_size = 21 # size of data --- unsure of what this should be for a set of points?? Possibly the output layer?
n_classes = 500 # since in reality, there is an infinite number of classes, this number is chosen arbitrarily

# Given: [xi, f(xi)] 
# Compute: d 
x = tf.placeholder('float', [None, data_size]) # (0, data_size) as an ordered pair
y = tf.placeholder('float') 

def neural_network_model(data):

	# (input_data * weights) + biases

	hidden_1_layer = {'weights': tf.Variable(tf.random_normal([data_size, n_nodes_hl1])),
					'biases': tf.Variable(tf.random_normal([n_nodes_hl1]))}
"""creates a variable --> weights = finds normal of the data_size and number of nodes for layer 1
						  biases = finds normal of the nodes for layer 1
"""
	hidden_2_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
					'biases': tf.Variable(tf.random_normal([n_nodes_hl2]))}
"""creates a variable --> weights = finds normal of the number of nodes for layer 1 and number of nodes for layer 2
						  biases = finds normal of the nodes for layer 2
"""	
	hidden_3_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
					'biases': tf.Variable(tf.random_normal([n_nodes_hl3]))}
"""creates a variable --> weights = finds normal of the number of nodes for layer 2 and number of nodes for layer 3
						  biases = finds normal of the nodes for layer 3
"""	
	output_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
					'biases': tf.Variable(tf.random_normal([n_classes]))}
"""creates a variable --> weights = finds normal of the number of nodes for layer 3 and number of classes
						  biases = finds normal of the number of classes
"""	
	l1 = tf.add(tf.matmul(data, hidden_1_layer['weights']), hidden_1_layer['biases'])
	"""relu = activation function
	   The activation ops provide different types of nonlinearities for use in neural networks.
	   continuous but not everywhere differentiable functions
	   returns a Tensor
	"""
	l1 = tf.nn.relu(l1, give_this_a_name)

	l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']), hidden_2_layer['biases'])
	# relu = activation function
	 = tf.nn.relu(l2, give_this_a_name)

	l3 = tf.add(tf.matmul(l2, hidden_3_layer['weights']), hidden_3_layer['biases'])
	# relu = activation function
	l3 = tf.nn.relu(l3, give_this_a_name)

	output = tf.matmul(l3, output_layer['weights']) + output_layer['biases']
	#(multiplies layer 3 with the weight of the output layer) + the output layer biases 
	
	return output

def train_nerual_network(x):
	predictor = neural_network_model(x) #(input_data * weights) + biases
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predictor, labels=y))
	# minimize our cost
	
	optimizer = tf.train.AdamOptimizer().minimize(cost) #the AdamOptimizer was chosen to optimize our data, but why?

	hm_epochs = 10 #epoch = an repetition of data training; chosen arbitrarily

	with tf.Session() as sess: # begin tf session... we will need to keep this open in order to keep our "machine"
		sess.run(tf.global_variables_initializer())

		for epoch in range(hm_epochs):
			epoch_loss = 0
			for _ in range(int(mnist.train.num_examples/data_points)): # INSERT DAN'S TRAINING SET HERE
								# 500,000 / 1,000 = 500 
				epoch_x, epoch_y = mnist.train.next_batch(data_points) #  INSERT DAN'S TRAINING SET HERE
				_, c = sess.run([optimizer, cost], feed_dict = {x: epoch_x, y: epoch_y})
				epoch_loss += c
			print('Epoch', epoch, 'completed out of', hm_epochs, 'loss:', epoch_loss) #tells the user how well the data was trained

		correct = tf.equal(tf.argmax(predictor, 1), tf.argmax(y, 1))

		accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

		print('Accuracy:', accuracy.eval({x: mnist.test.images, y: mnist.test.labels})) #
		return predictor

machine = train_nerual_network(x) #created a stored "machine" to train the nerual net consistantly

#use this machine to test other data sets in order to classify the class of a polynomial 
