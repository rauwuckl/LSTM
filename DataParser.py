import os
import re
import pickle
from collections import Counter, defaultdict

class DataParser:

    replaceTokens = {'.':' <dot> ',
                     ',': ' <comma> ',
                     '?': ' <question> ',
                     '!': ' <exclamation> ',
                     '"': ' <quotes> ',
                     '(': ' <brOben> ',
                     ')': ' <brClose> ',
                     '\'s': ' <tokenS> ',
                     '<br />': ' '}


    def __init__(self, path):
        if(os.path.isdir(path)):
            self.path = path
        else:
            raise Exception('The File path does not exit')

        self.loadTestData()
        self.loadTrainingData()

        self.makeDict()

        self.convertAllToIds(self.trainingData)
        self.convertAllToIds(self.testData)


    def saveAll(self):

        allData = {'trainData': self.trainingData, 'testData': self.testData, 'word2id': self.word2id, 'id2word': self.id2word}

        with open('imdbData.pickle', 'wb') as f:
            pickle.dump(allData, f)


    def loadFiles(self, folder, label):
        newpath = self.path + '/' + folder
        if not os.path.isdir(newpath):
            raise Exception('The path does not exist: {}'.format(newpath))

        result = []

        for file in os.listdir(newpath):
            m = re.match(r"(\d*)_(\d*).txt", file)
            id, rating = m.groups()
            #print("{}: id={}, rating={}".format(file, id, rating))
            with open(newpath+'/' + file, 'r') as f:
                text = f.read()
                result.append({'text': text, 'rating':rating, 'label': label, 'id':int(id)})

        return self.cleanList(result)

    @classmethod
    def tokenize(cls, string):
        string = string.lower()
        for token in DataParser.replaceTokens.keys():
            string = string.replace(token, DataParser.replaceTokens[token])
        return string.split()

    def cleanList(self, list):
        for review in list:
            review['text'] = self.tokenize(review['text'])
        return list

    def loadTrainingData(self):
        pos = self.loadFiles('train/pos', label=1)
        neg = self.loadFiles('train/neg', label=0)

        self.trainingData = pos + neg

    def loadTestData(self):
        pos = self.loadFiles('test/pos', label=1)
        neg = self.loadFiles('test/neg', label=0)

        self.testData = pos + neg


    def makeDict(self, vocabSize=60000):
        counter = Counter()
        allData = self.trainingData + self.testData

        for review in allData:
            counter.update(review['text'])

        mostCommon = counter.most_common(vocabSize)
        emptyTokenId = len(mostCommon) #will be used for the rare words
        self.word2id = {}
        for i, word in enumerate(mostCommon):
            self.word2id[word[0]] = i

        self.id2word = list(zip(*mostCommon))[0] + ('<unknownToken>',)

        #self.id2word.append('<unknownToken>')

    def convertAllToIds(self, listOfReviews):
        for review in listOfReviews:
            review['text'] = self.convertToIds(review['text'])

    def convertToIds(self, textlist):
        # len(self.word2id) so that unknown words return the number one after the last proper one
        return list(map(lambda w: self.word2id.get(w, len(self.word2id)), textlist))


    def convertToWords(self, idlist):
        return list(map(lambda i: self.id2word[i], idlist))

