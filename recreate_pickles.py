# -*- coding: utf-8 -*-
import cPickle
from nltk.corpus import stopwords
from nltk.classify import NaiveBayesClassifier
import nltk.data 
from nltk.corpus import movie_reviews
import re
from nltk.tag import pos_tag

STOP_WORDS = cPickle.load(open('stopwords.pickle'))
# Strip urls
URL_REGEX = re.compile(r'''(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’]))''',re.I) #http://daringfireball.net/2010/07/improved_regex_for_matching_urls  

#Strip punctuations and hash tags 
PUNCT_REGEX = re.compile(r'\,|\;|\#',re.I)

#Strip use reference
USER_REGEX = re.compile(r'@\w+?',re.I) 


TRAIN_DATASET_LOC = 'corpora/tweets_train'
TEST_DATASET_LOC = 'corpora/tweets_test'

POS_TAGS={}

SMILIES = (':)',':-)',':D',':-D',';D',';)',';-)',';-D',':(',':-(',';(',';-(',':O')

ALLOWED_POS_TAGS = ('JJ', 'JJR','JJS', 'RB', 'RBR', 'RBS') 

def __getPosTags(word):
    if not POS_TAGS.has_key(word):
        POS_TAGS[word]=nltk.tag.pos_tag([word])[0][1]
    return POS_TAGS[word]
    
def __word_feats(words):
    return dict([(word.lower(), True) for word in words])
  
def __word_feats_pos(words):
    return dict([(word.lower(), True) for word in words if word not in STOP_WORDS or word in SMILIES \
                     or __getPosTags(word) in ('JJ', 'JJR','JJS', 'RB', 'RBR', 'RBS')])
        
def __word_feats_neg(words):
    #return __word_feats(words)
    return dict([(word.lower(), True) for word in words if word not in STOP_WORDS])

def create_stopwords():
    print "Recreating stop word pickles."
    file_name = 'stopwords.pickle'
    words = stopwords.words()
    __write_file(file_name,cPickle.dumps(words))
    print "Done!"

def __write_file(file_name,pickle_dump):
    f=open(file_name,'w')
    f.write(pickle_dump)
    f.close()

def create_test_pickles():
    print "Recreating test data pickles"
    test_dir = nltk.data.find(TEST_DATASET_LOC)
    test_data = nltk.corpus.CategorizedPlaintextCorpusReader(test_dir, fileids='.*\.txt',cat_pattern="(pos|neg)")


    negids = test_data.fileids('neg')
    posids = test_data.fileids('pos')

    negfeats = [(__word_feats(test_data.words(fileids=[f])), 'neg') for f in negids]
    posfeats = [(__word_feats(test_data.words(fileids=[f])), 'pos') for f in posids]
    
    pos_file_name = 'positive_test.pickle'
    neg_file_name = 'negative_test.pickle'

    __write_file(pos_file_name,cPickle.dumps(posfeats))
    __write_file(neg_file_name,cPickle.dumps(negfeats))
    print "Done!"

def create_train_classifier():
    print "Recreating training classifier"
    corpus_dir = nltk.data.find(TRAIN_DATASET_LOC)
    train_data = nltk.corpus.CategorizedPlaintextCorpusReader(corpus_dir, fileids='.*\.txt',cat_pattern="(pos|neg)")
        

    negids_train = train_data.fileids('neg')
    posids_train = train_data.fileids('pos')
        
    negids_movies = movie_reviews.fileids('neg')
    posids_movies = movie_reviews.fileids('pos')

    negfeats = [(__word_feats_neg(train_data.words(fileids=[f])), 'neg') for f in negids_train]
    posfeats = [(__word_feats_pos(train_data.words(fileids=[f])), 'pos') for f in posids_train]

    negfeats.extend([(__word_feats_neg(movie_reviews.words(fileids=[f])), 'neg') for f in negids_movies])
    posfeats.extend([(__word_feats_pos(movie_reviews.words(fileids=[f])), 'pos') for f in posids_movies])

    trainfeats = negfeats + posfeats

    classifier = NaiveBayesClassifier.train(trainfeats)
    
    pos_file_name = 'positive_train.pickle'
    neg_file_name = 'negative_train.pickle'
    class_file_name = 'nbClassifier.pickle'
    
    __write_file(pos_file_name,cPickle.dumps(posfeats))
    __write_file(neg_file_name,cPickle.dumps(negfeats))
    __write_file(class_file_name,cPickle.dumps(classifier))
    print "Done!"

if __name__ == '__main__':
    #TODO add Option parse
    import sys
    #Make sure to create stop_words before anything else.
    if 'stop_words' in sys.argv[1:]:
        create_stopwords()

    if 'classifier' in sys.argv[1:]:
        create_train_classifier()
    if 'test_pickles' in sys.argv[1:]:
        create_test_pickles()    
    
