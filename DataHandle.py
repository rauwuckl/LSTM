import pickle
import numpy as np
from collections import Counter

class DataHandle:

    def __init__(self, dataFileName):
        print('passilove zousnobu')
        with open(dataFileName, 'rb') as f:
            data = pickle.load(f)

        self.trainingData = data['trainData']
        self.testData = data['testData']
        self.word2id = data['word2id']
        self.id2word = data['id2word']

        firstUnusedID = len(self.word2id)

        self.specialIDs = {'unkown': firstUnusedID, 'fillerToken': firstUnusedID+1}
        self.id2word += ('<fillerToken>',)

    def convertToIds(self, textlist):
        # len(self.word2id) so that unknown words return the number one after the last proper one
        return list(map(lambda w: self.word2id.get(w, len(self.word2id)), textlist))

    def convertToWords(self, idlist):
        return list(map(lambda i: self.id2word[i], idlist))


    @property
    def vocabSize(self):
        return len(self.id2word)

    def countLabels(self, data):
        Counter([w['label'] for w in data])


    def epoch(self, batchSize, seqLength, testData=False):
        if testData:
            groundData = self.testData
        else:
            groundData = self.trainingData


        indices = np.random.permutation(len(self.trainingData))

        N_batches = len(groundData)//batchSize

        for i in range(N_batches):
            data, label = self.getBatchData(indices[i*batchSize:(i+1)*batchSize], groundData)
            yield (self.splitMatrix(data, seqLength), label)

    def getBatchData(self, indices, groundData):
        '''returns matrix of shape [batchsize x maxLength of sequence in indices] where batchsize is len(indices)
        everything else is filled up with the value of the filler token'''
        data = []
        label = []
        # maxLen = 0
        dataLength = np.zeros((len(indices),),dtype=int)
        for indx, i in enumerate(indices):
            # maxLen = max(maxLen, len(groundData[i]['text']))
            dataLength[indx] = len(groundData[i]['text'])
            data.append(groundData[i]['text'])
            label.append(groundData[i]['label'])
        print("dataLengths: {}".format(dataLength))
        maxLen = np.max(dataLength)

        #make matrix batchSize x maxLen filled with the filler id, chop it up later
        outputMatrix = np.ones((len(indices), maxLen), dtype=np.int32)*self.specialIDs['fillerToken']

        for i,d in enumerate(data):
            outputMatrix[i, 0:len(d)] = d

        return outputMatrix, np.array(label)

    def splitMatrix(self, matrix, seqLength):
        batchSize, totalLength = matrix.shape
        #nSplitPoints = totalLength//seqLength + 1
        splitIndices = range(seqLength, totalLength, seqLength)
        listOfSubMatrices = np.split(matrix, splitIndices, axis=1)
        del listOfSubMatrices[-1]
        return listOfSubMatrices[:10]


    def getSequences(self, seqLength, maxLength):
        pass