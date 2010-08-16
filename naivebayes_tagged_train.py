# -*- coding: utf-8 -*-
from naivebayes_train import NBSentimentClassifier

class NBTaggedSentimentClassifier(NBSentimentClassifier):
    def populate_config(self):
        ##Cheap way of handling different configurations:
        self.PICKLE_FILES_DICT = {'classifier':'nbClassifier.pickle.tagged',
                                  'positive_train':'positive_train.pickle.tagged',
                                  'negative_train': 'negative_train.pickle.tagged'}


if __name__ == '__main__':
    nbObj = NBTaggedSentimentClassifier()
    nbObj.run_test_pos()
    nbObj.run_test_neg()
    nbObj.run_test_all()
    nbObj.show_overall_stats(nbObj.posfeats+nbObj.negfeats)
