# -*- coding: utf-8 -*-
import collections
import nltk.metrics
import nltk.classify.util
import cPickle as pickle

class NBSentimentClassifier(object):
    
    def __init__(self):
        #Train the Model
        self.train_data()
        self.__create_test_feats()
    
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

        print "Total Documents:%d Positive Documents:%d(%d) Negative Documents:%d(%d)"\
            %(len(refsets['pos'])+len(refsets['neg']),len(testsets['pos']),len(self.posfeats),len(testsets['neg']),len(self.negfeats))

 
    def train_data(self):
        self.classifier = pickle.load(open('nbClassifier.pickle'))

    def __create_test_feats(self):
        self.posfeats = pickle.load(open('positive_test.pickle'))
        self.negfeats = pickle.load(open('negative_test.pickle'))

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
