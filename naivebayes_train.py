# -*- coding: utf-8 -*-
import collections
import nltk.metrics
import nltk.classify.util
import cPickle as pickle
from nltk.classify import NaiveBayesClassifier
class NBSentimentClassifier(object):
    
    def __init__(self):
        #Train the Model
        self.populate_config()
        self.load_feats()
        self.train_data()
        self.test_data()
    
    def populate_config(self):
        ##Cheap way of handling different configurations:
        self.PICKLE_FILES_DICT = {'classifier':'nbClassifier.pickle',
                                  'positive_train':'positive_train.pickle',
                                  'negative_train': 'negative_train.pickle'}


    def run_test_pos(self):
        print "Testing positive feature sets alone."
        self.run_test(self.posfeats)
        
    def run_test_neg(self):
        print "Testing negative feature sets alone."
        self.run_test(self.negfeats)
        

    def run_test_all(self):
        print "Testing all feature sets."
        self.run_test(self.posfeats + self.negfeats)

    def show_overall_stats(self,testfeats):
        print "Printing all stats."
        refsets =  collections.defaultdict(set)
        testsets = collections.defaultdict(set)
        
        for i, (feats, label) in enumerate(testfeats):
            refsets[label].add(i)
            observed = self.classifier.classify(feats)
            testsets[observed].add(i)
            
        print 'pos precision:', nltk.metrics.precision(refsets['pos'], testsets['pos'])
        print 'pos recall:', nltk.metrics.recall(refsets['pos'], testsets['pos'])
        print 'pos F-measure:', nltk.metrics.f_measure(refsets['pos'], testsets['pos'])
        print 'neg precision:', nltk.metrics.precision(refsets['neg'], testsets['neg'])
        print 'neg recall:', nltk.metrics.recall(refsets['neg'], testsets['neg'])
        print 'neg F-measure:', nltk.metrics.f_measure(refsets['neg'], testsets['neg'])

        print "Documents- Total:%d Positive:%d(%d) Negative:%d(%d) Unclassified:%d"\
            %(len(refsets['pos'])+len(refsets['neg']),len(testsets['pos']),\
                  len(self.posfeats),len(testsets['neg']),len(self.negfeats),\
                  len(self.posfeats)+len(self.negfeats)-len(testsets['pos'])-len(testsets['neg']))

 
    def train_data(self):
        self.classifier = NaiveBayesClassifier.train(self.all_posfeats[:int(0.75*len(self.all_posfeats))]+\
                                                         self.all_negfeats[:int(0.75*len(self.all_negfeats))])
        
    def load_feats(self):
        self.all_posfeats = pickle.load(open(self.PICKLE_FILES_DICT['positive_train']))
        self.all_negfeats = pickle.load(open(self.PICKLE_FILES_DICT['negative_train']))
    
    def test_data(self):
        self.posfeats = self.all_posfeats[int(0.75*len(self.all_posfeats)):]
        self.negfeats = self.all_negfeats[int(0.75*len(self.all_negfeats)):]

    def run_test(self,testfeats):
        accuracy = nltk.classify.util.accuracy(self.classifier, testfeats)
        print 'Tested on %d instances, Accuracy: %f' % (len(testfeats),accuracy)
        print "================================================================="
        print ""

if __name__ == '__main__':
    nbObj = NBSentimentClassifier()
    nbObj.run_test_pos()
    nbObj.run_test_neg()
    nbObj.run_test_all()
    nbObj.show_overall_stats(nbObj.posfeats+nbObj.negfeats)
