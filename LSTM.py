import tensorflow as tf
from DataHandle import DataHandle
from matplotlib import pyplot as plt
import numpy as np

FLAGS = type('FLAGS', (), {})
FLAGS.embeddingSize = 200
FLAGS.memorySize = 200
FLAGS.batchSize = 30
FLAGS.seqLength = 13
FLAGS.stdActFun = tf.nn.sigmoid
FLAGS.crossEntropy = tf.nn.sparse_softmax_cross_entropy_with_logits
FLAGS.optimizer = tf.train.AdamOptimizer
FLAGS.learningRate = 5.01
FLAGS.nEpochs = 2
FLAGS.plotEveryN = 10
FLAGS.saveEveryN = 1000

class LSTM:

    def __init__(self, dataHandle):
        tf.reset_default_graph()
        self.data_handle = dataHandle

        self.buildNetwork()
        self.buildStateResetOP()

    def getEmbeding(self):
        with tf.variable_scope("embedding"):
            embeddings = tf.get_variable("embed", [self.data_handle.vocabSize, FLAGS.embeddingSize], tf.float32, tf.random_uniform_initializer(-1.0, 1.0))

        # norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
        # normalized_embeddings = embeddings / norm
        #normalizedEmbed = tf.nn.embedding_lookup((normalized_embeddings, wordIds))

        return embeddings

    def fullyLayer(self, input, outputSize, name, actFunction=FLAGS.stdActFun):
        #print("bla: {}".format(input.get_shape()))
        _nExamples, nNeuronsInput = input.get_shape()
        with tf.variable_scope(name):
            weights = tf.get_variable("weights", shape=[nNeuronsInput, outputSize],
                                      initializer=tf.random_normal_initializer(stddev=0.05))
            bias = tf.get_variable("bias", shape=[outputSize], initializer=tf.constant_initializer(0.0))

            internalActivation = tf.matmul(input, weights)
            result = actFunction(internalActivation + bias)
        print("fullyLayer: {}, shape: {} --> {}".format(name, input.get_shape(), result.get_shape()))
        return result

    def buildStateResetOP(self):
        self.resetStateOP = self.state.assign(self.lstmCell.zero_state(FLAGS.batchSize, tf.float32))
        print(self.resetStateOP)

    def buildNetwork(self):
        self.input = tf.placeholder(tf.int64, shape=[FLAGS.batchSize, FLAGS.seqLength])
        self.desired = tf.placeholder(tf.int64, shape=[FLAGS.batchSize, ])

        self.embedding = self.getEmbeding()

        self.embeded = tf.nn.embedding_lookup(self.embedding, self.input)

        print(self.embeded.get_shape())

        self.lstmCell = tf.contrib.rnn.BasicLSTMCell(FLAGS.memorySize, state_is_tuple=False)

        zeroState = self.lstmCell.zero_state(FLAGS.batchSize, tf.float32)
        self.state = tf.Variable(zeroState, trainable=False)

        outputsUnasigned, newState = tf.nn.dynamic_rnn(self.lstmCell, self.embeded, initial_state=self.state)

        with tf.control_dependencies([self.state.assign(newState)]):
            self.outputLSTM = tf.identity(outputsUnasigned)


        print("output: {}, state: {}".format(self.outputLSTM.get_shape(), self.state.get_shape()))

        meanedLSTMOutput = tf.reduce_mean(self.outputLSTM, axis=1)

        print("meanedLSTMOutput: {}".format(meanedLSTMOutput.shape))

        self.afterFull = self.fullyLayer(meanedLSTMOutput, 2, 'readout')

        crossEnt = FLAGS.crossEntropy(logits=self.afterFull, labels=self.desired)
        self.crossEntropy = tf.reduce_mean(crossEnt)

        self.optimizer = FLAGS.optimizer(learning_rate=FLAGS.learningRate)

        self.trainingStep = self.optimizer.minimize(self.crossEntropy)

        self.prediction = tf.argmax(self.afterFull, 1)
        self.nCorrect = tf.equal(self.prediction, self.desired)
        # accuracy = tf.equal(tf.argmax(output_ron_2,1), tf.argmax(desired,1))
        self.accuracy = tf.reduce_mean(tf.cast(self.nCorrect, tf.float32))

        print("afterFull: {}, crossEnt: {}".format(self.afterFull.get_shape(), self.crossEntropy.get_shape()))



data_handle = DataHandle('imdbData.pickle')
lstm = LSTM(data_handle)
# lstm = LSTM(type('DataHandle', (), {'vocabSize':60000}))



savePath = 'saves/v1.cpkt'
restorePath = 'saves/v1.cpkt'
restorePath = None
# savePath = None

plt.ion()
entropyFig, entropyAx = plt.subplots(1, 1)


plotWindow = 1000

i_current_batch = 0
losses = np.zeros(plotWindow)

with tf.Session() as session:
    train_writer = tf.summary.FileWriter('summary/bla', session.graph)

    saver = tf.train.Saver()
    if (restorePath is None):
        session.run(tf.global_variables_initializer())
        print("initialized variables")
    else:
        saver.restore(session, restorePath)
        print("restored variables")


    for current_epoch in range(FLAGS.nEpochs):
        for batch_data in data_handle.epoch(FLAGS.batchSize, FLAGS.seqLength):
            sequences, label = batch_data

            #reset hidden state
            session.run(lstm.resetStateOP)
            print('new Sequences')
            for subSeq in sequences:
                _, lossvalue, debugVal = session.run([lstm.trainingStep, lstm.crossEntropy, lstm.afterFull], feed_dict={lstm.input: subSeq, lstm.desired: label})
                losses[i_current_batch] = lossvalue

                print("debugVal: {}".format(debugVal))
                # print("{} <- label".format(label))

                if(i_current_batch % FLAGS.plotEveryN == 0):
                    entropyAx.cla()
                    entropyAx.plot(losses)
                    entropyFig.canvas.draw()
                    plt.pause(0.0001)

                i_current_batch+=1

                if(i_current_batch % plotWindow == 0):
                    #the next i would be out of bounds for losses. so we need to append another window
                    losses = np.concatenate([losses, np.zeros(plotWindow)])

                if(i_current_batch % FLAGS.saveEveryN == 0):

                    if (not savePath is None):
                        path = saver.save(session, savePath)
                        print("path: {}".format(path))



def plot_with_labels(low_dim_embs, labels, filename='tsne.png'):
  assert low_dim_embs.shape[0] >= len(labels), "More labels than embeddings"
  plt.figure(figsize=(18, 18))  # in inches
  for i, label in enumerate(labels):
    x, y = low_dim_embs[i, :]
    plt.scatter(x, y)
    plt.annotate(label,
                 xy=(x, y),
                 xytext=(5, 2),
                 textcoords='offset points',
                 ha='right',
                 va='bottom')

  plt.savefig(filename)


from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
plot_only = 500
low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only, :])
labels = [data_handle.most_common[i][0] for i in range(plot_only)]
plot_with_labels(low_dim_embs, labels)