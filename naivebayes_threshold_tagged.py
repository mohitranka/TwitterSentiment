# -*- coding: utf-8 -*-
from naivebayes_threshold import NBThresholdClassifier
class NBThresholdTaggedClassifier(NBThresholdClassifier):        
    def populate_config(self):
        ##Cheap way of handling different configurations:
        self.PICKLE_FILES_DICT = {'classifier':'nbClassifier.pickle.tagged',
                                  'positive_test':'positive_test.pickle.tagged',
                                  'negative_test': 'negative_test.pickle.tagged'}


if __name__ == '__main__':
    nbObj = NBThresholdTaggedClassifier()
    nbObj.run_test_pos()
    nbObj.run_test_neg()
    nbObj.run_test_all()
    nbObj.show_overall_stats(nbObj.get_in_threshold_features(nbObj.posfeats+nbObj.negfeats))
