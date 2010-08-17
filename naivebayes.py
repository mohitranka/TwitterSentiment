# -*- coding: utf-8 -*-
import collections
import nltk.metrics
import nltk.classify.util
import cPickle
import os

MIN_THRESHOLD = 0.37
MAX_THRESHOLD = 0.63

class NBSentimentClassifier(object):     
    def __init__(self):
        #Train the Model
        self.results = ""
        self.load_train_data()
        self.load_test_data()
     
    def get_in_threshold_features(self,features):
        in_range_features = []
        for feat in features:
            if not(MIN_THRESHOLD <= self.classifier.prob_classify(feat[0]).prob('pos') <= MAX_THRESHOLD):
                in_range_features.append(feat)
        return in_range_features

    def run_test_pos(self):
        s = "Testing positive feature sets alone.\n"
        s+=self.run_test(self.posfeats)
        print s
        self.results+=s 

    def run_test_neg(self):
        s="Testing negative feature sets alone.\n"
        s+=self.run_test(self.negfeats)
        print s
        self.results+=s

    def run_test_all(self):
        s="Testing all feature sets.\n"
        s+=self.run_test(self.posfeats + self.negfeats)
        print s
        self.results+=s

    def show_overall_stats(self,testfeats):
        s="Printing all stats.\n"
        refsets =  collections.defaultdict(set)
        testsets = collections.defaultdict(set)
        
        for i, (feats, label) in enumerate(self.get_in_threshold_features(testfeats)):
            refsets[label].add(i)
            observed = self.classifier.classify(feats)
            testsets[observed].add(i)
            
        s+= 'pos precision:%f'%(nltk.metrics.precision(refsets['pos'], testsets['pos']))
        s+='\npos recall:%f'%(nltk.metrics.recall(refsets['pos'], testsets['pos']))
        s+='\npos F-measure:%f'%(nltk.metrics.f_measure(refsets['pos'], testsets['pos']))
        s+='\nneg precision:%f'%(nltk.metrics.precision(refsets['neg'], testsets['neg']))
        s+='\nneg recall:%f'%(nltk.metrics.recall(refsets['neg'], testsets['neg']))
        s+='\nneg F-measure:%f'%(nltk.metrics.f_measure(refsets['neg'], testsets['neg']))

        s+="\nDocuments- Total:%d Positive:%d(%d) Negative:%d(%d) Unclassified:%d"\
            %(len(refsets['pos'])+len(refsets['neg']),len(testsets['pos']),\
                  len(self.posfeats),len(testsets['neg']),len(self.negfeats),\
                  len(self.posfeats)+len(self.negfeats)-len(testsets['pos'])-len(testsets['neg']))
        print s
        self.results+=s
 
    def load_train_data(self):
        self.classifier = cPickle.load(open('pickles'+os.sep+'nbClassifier.pickle'))
        
    def load_test_data(self):
        self.posfeats = cPickle.load(open('pickles'+os.sep+ 'positive_test.pickle'))
        self.negfeats = cPickle.load(open('pickles'+os.sep+'negative_test.pickle'))

    def run_test(self,testfeats):
        valid_feats=self.get_in_threshold_features(testfeats)
        accuracy = nltk.classify.util.accuracy(self.classifier, valid_feats)
        s='Tested on %d instances, Accuracy: %f' % (len(testfeats),accuracy)
        s+="\n=================================================================\n"
        return s

    def write_results(self,result_file):
        f = open(result_file,'w')
        f.write(self.results)
        f.close()

if __name__ == '__main__':
    nbObj = NBSentimentClassifier()
    nbObj.run_test_pos()
    nbObj.run_test_neg()
    nbObj.run_test_all()
    nbObj.show_overall_stats(nbObj.posfeats+nbObj.negfeats)
    nbObj.write_results('results' + os.sep + 'naivebayes.results')
