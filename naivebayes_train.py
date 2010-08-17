# -*- coding: utf-8 -*-
import cPickle
from nltk.classify import NaiveBayesClassifier
from naivebayes import NBSentimentClassifier
import os
import sys

class NBTrainSentimentClassifier(NBSentimentClassifier):
    
    def __init__(self):
        #Train the Model
        self.results = ""
        self.TRAIN_FACTOR = 0.75
        if len(sys.argv)>1:
            self.TRAIN_FACTOR = float(sys.argv[1])
        s = "Training with %f\n"%(self.TRAIN_FACTOR)
        print s
        self.results+=s
        self.load_feats()
        self.load_train_data()
        self.load_test_data()
        #super(NBSentimentClassifier,self).__init__()
    
    def load_train_data(self):
        self.classifier = NaiveBayesClassifier.train(self.all_posfeats[:int(self.TRAIN_FACTOR*len(self.all_posfeats))]+\
                                                         self.all_negfeats[:int(self.TRAIN_FACTOR*len(self.all_negfeats))])
        
    def load_feats(self):
        self.all_posfeats = cPickle.load(open( 'pickles'+os.sep+'positive_train.pickle'))
        self.all_negfeats = cPickle.load(open( 'pickles'+os.sep+'negative_train.pickle'))
    
    def load_test_data(self):
        self.posfeats = self.all_posfeats[int(self.TRAIN_FACTOR*len(self.all_posfeats)):]
        self.negfeats = self.all_negfeats[int(self.TRAIN_FACTOR*len(self.all_negfeats)):]


if __name__ == '__main__':
    nbObj = NBTrainSentimentClassifier()
    nbObj.run_test_pos()
    nbObj.run_test_neg()
    nbObj.run_test_all()
    nbObj.show_overall_stats(nbObj.posfeats+nbObj.negfeats)
    nbObj.write_results('results' + os.sep + 'naivebayes_train.results')
        
