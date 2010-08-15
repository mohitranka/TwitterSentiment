# -*- coding: utf-8 -*-
import collections
import nltk.metrics
import nltk.classify.util
from nltk.classify import NaiveBayesClassifier
import nltk.data
from nltk.corpus import stopwords as sw 
from nltk.corpus import movie_reviews
import re
import cPickle as pickle
#STOP_WORDS = sw.words()

STOP_WORDS = pickle.load(open('stopwords.pickle'))
# Strip urls
URL_REGEX = re.compile(r'''(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’]))''',re.I) #http://daringfireball.net/2010/07/improved_regex_for_matching_urls  

#Strip punctuations and hash tags 
PUNCT_REGEX = re.compile(r'\,|\;|\#',re.I)

#Strip use reference
USER_REGEX = re.compile(r'@\w+?',re.I) 


#TRAIN_DATASET_LOC = 'corpora/tweets_train'
#TEST_DATASET_LOC = 'corpora/tweets_test'

class NBSentimentClassifier:
    
    def __init__(self):
        #Train the Model
        self.__train_data()
        self.__create_test_feats()
    
    def run_test_pos(self):
        print "Testing positive feature sets alone."
        self.__run_test(self.posfeats)
        
    def run_test_neg(self):
        print "Testing negative feature sets alone."
        self.__run_test(self.negfeats)
        

    def run_test_all(self):
        print "Testing all feature sets."
        self.__run_test(self.posfeats + self.negfeats)
        refsets =  collections.defaultdict(set)
        testsets = collections.defaultdict(set)
         
        for i, (feats, label) in enumerate(self.posfeats + self.negfeats):
            refsets[label].add(i)
            observed = self.classifier.classify(feats)
            testsets[observed].add(i)
            
        print 'pos precision:', nltk.metrics.precision(refsets['pos'], testsets['pos'])
        print 'pos recall:', nltk.metrics.recall(refsets['pos'], testsets['pos'])
        print 'pos F-measure:', nltk.metrics.f_measure(refsets['pos'], testsets['pos'])
        print 'neg precision:', nltk.metrics.precision(refsets['neg'], testsets['neg'])
        print 'neg recall:', nltk.metrics.recall(refsets['neg'], testsets['neg'])
        print 'neg F-measure:', nltk.metrics.f_measure(refsets['neg'], testsets['neg'])


    def __word_feats(self,words):
	#words = [word.lower() for word in words if word not in STOP_WORDS]
        #words = [word for word in words if not URL_REGEX.match(word)]
        #words = [word for word in words if not PUNCT_REGEX.match(word)]
        #words = [word for word in words if not USER_REGEX.match(word)]
	return dict([(word.lower(), True) for word in words])
    
    def __word_feats_pos(self,words):
        #words = [word.lower() for word in words if word not in STOP_WORDS]
        #words = [word for word in words if not URL_REGEX.match(word)]
        #words = [word for word in words if not PUNCT_REGEX.match(word)]
        #words = [word for word in words if not USER_REGEX.match(word)]
        return dict([(word.lower(), True) for word in words])
        
    def __word_feats_neg(self,words):
        #words = [word.lower() for word in words if word not in STOP_WORDS]
        #words = [word for word in words if not URL_REGEX.match(word)]
        #words = [word for word in words if not PUNCT_REGEX.match(word)]
        #words = [word for word in words if not USER_REGEX.match(word)]
        return dict([(word.lower(), True) for word in words])

    def __train_data(self):
        self.classifier = pickle.load(open('nbClassifier.pickle'))
        # corpus_dir = nltk.data.find(TRAIN_DATASET_LOC)
        # train_data = nltk.corpus.CategorizedPlaintextCorpusReader(corpus_dir, fileids='.*\.txt',cat_pattern="(pos|neg)")
        

        # negids_train = train_data.fileids('neg')
        # posids_train = train_data.fileids('pos')
        
        # negids_movies = movie_reviews.fileids('neg')
        # posids_movies = movie_reviews.fileids('pos')

        # negfeats = [(self.__word_feats_neg(train_data.words(fileids=[f])), 'neg') for f in negids_train]
        # posfeats = [(self.__word_feats_pos(train_data.words(fileids=[f])), 'pos') for f in posids_train]

        # negfeats.extend([(self.__word_feats_neg(movie_reviews.words(fileids=[f])), 'neg') for f in negids_movies])
        # posfeats.extend([(self.__word_feats_pos(movie_reviews.words(fileids=[f])), 'pos') for f in posids_movies])

        # trainfeats = negfeats + posfeats

        # self.classifier = NaiveBayesClassifier.train(trainfeats)
        # self.classifier.show_most_informative_features()
        # print "Trained on %d instances"%(len(trainfeats))

    def __create_test_feats(self):
        # test_dir = nltk.data.find(TEST_DATASET_LOC)
        # test_data = nltk.corpus.CategorizedPlaintextCorpusReader(test_dir, fileids='.*\.txt',cat_pattern="(pos|neg)")


        # negids = test_data.fileids('neg')
        # posids = test_data.fileids('pos')

        # self.negfeats = [(self.__word_feats(test_data.words(fileids=[f])), 'neg') for f in negids]
        # self.posfeats = [(self.__word_feats(test_data.words(fileids=[f])), 'pos') for f in posids]
        self.posfeats = pickle.load(open('positive_test.pickle'))
        self.negfeats = pickle.load(open('negative_test.pickle'))

    def __run_test(self,testfeats):
        accuracy = nltk.classify.util.accuracy(self.classifier, testfeats)
        print 'Tested on %d instances, Accuracy: %f' % (len(testfeats),accuracy)
        print "================================================================="
        print ""

if __name__ == '__main__':
    nbObj = NBSentimentClassifier()
    nbObj.run_test_pos()
    nbObj.run_test_neg()
    nbObj.run_test_all()
