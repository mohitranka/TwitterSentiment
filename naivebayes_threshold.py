# -*- coding: utf-8 -*-
import nltk.classify.util
from naivebayes import NBSentimentClassifier
from nltk.classify import NaiveBayesClassifier
import cPickle as pickle

MIN_THRESHOLD = 0.37
MAX_THRESHOLD = 0.63

class NBThresholdClassifier(NBSentimentClassifier):        
    def populate_config(self):
        ##Cheap way of handling different configurations:
        self.PICKLE_FILES_DICT = {'classifier':'nbClassifier.pickle',
                                  'positive_test':'positive_test.pickle',
                                  'negative_test': 'negative_test.pickle'}

    def run_test(self,testfeats):
        #Discard for probabilities in MIN_THRESHOLD to MAX_THRESHOLD range, to increase the precision.
        valid_feats=self.get_in_threshold_features(testfeats)
        accuracy = nltk.classify.util.accuracy(self.classifier, valid_feats)
        print 'Tested on %d instances, Accuracy: %f' % (len(valid_feats),accuracy)
        print "================================================================="
        print ""

    # def train_data(self):
    #     super(NBThresholdClassifier, self).train_data()
    #     ##Retrain the model to disregrad the posts from MIN_THRESHOLD to MAX_THRESHOLD_RANGE 
    #     posfeats = pickle.load(open('positive_train.pickle'))
    #     negfeats = pickle.load(open('negative_train.pickle'))
    #     valid_feats = self.get_in_threshold_features(posfeats+negfeats)
    #     self.classifier = NaiveBayesClassifier.train(valid_feats)
        
    def get_in_threshold_features(self,features):
        in_range_features = []
        for feat in features:
            if not(MIN_THRESHOLD <= self.classifier.prob_classify(feat[0]).prob('pos') <= MAX_THRESHOLD):
                in_range_features.append(feat)
        return in_range_features
        

if __name__ == '__main__':
    nbObj = NBThresholdClassifier()
    nbObj.run_test_pos()
    nbObj.run_test_neg()
    nbObj.run_test_all()
    nbObj.show_overall_stats(nbObj.get_in_threshold_features(nbObj.posfeats+nbObj.negfeats))
