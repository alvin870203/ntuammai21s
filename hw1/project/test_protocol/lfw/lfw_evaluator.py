"""
@author: Haoran Jiang, Jun Wang
@date: 20201013
@contact: jun21wangustc@gmail.com
"""

import os
import sys
import numpy as np
    
class LFWEvaluator(object):
    """Implementation of LFW test protocal.
    
    Attributes:
        data_loader(object): a test data loader.
        pair_list(list): the pair list given by PairsParser.
        feature_extractor(object): a feature extractor.
    """
    def __init__(self, data_loader, pairs_parser_factory, feature_extractor, num_pair_per_subset=20, args_mode=''):
        """Init LFWEvaluator.

        Args:
            data_loader(object): a test data loader. 
            pairs_parser_factory(object): factory to produce the parser to parse test pairs list.
            pair_list(list): the pair list given by PairsParser.
            feature_extractor(object): a feature extractor.
        """
        self.data_loader = data_loader
        pairs_parser = pairs_parser_factory.get_parser()
        self.pair_list = pairs_parser.parse_pairs()
        self.feature_extractor = feature_extractor
        self.num_pair_per_subset = num_pair_per_subset # how many images per subset
        self.num_subset = len(self.pair_list) // self.num_pair_per_subset # how many subsets
        self.args_mode = args_mode
        self.result_or_not = ''

    def test(self, model, result_or_not):
        self.result_or_not = result_or_not
        image_name2feature = self.feature_extractor.extract_online(model, self.data_loader)
        mean, std = self.test_one_model(self.pair_list, image_name2feature)
        return mean, std

    def test_one_model(self, test_pair_list, image_name2feature, is_normalize = True):
        """Get the accuracy of a model.
        
        Args:
            test_pair_list(list): the pair list given by PairsParser. 
            image_name2feature(dict): the map of image name and it's feature.
            is_normalize(bool): wether the feature is normalized.

        Returns:
            mean: estimated mean accuracy.
            std: standard error of the mean.
        """
        # subsets_score_list = np.zeros((self.num_subset, self.num_pair_per_subset), dtype = np.float32)
        # subsets_label_list = np.zeros((self.num_subset, self.num_pair_per_subset), dtype = np.int8)
        subsets_score_list = np.zeros((len(self.pair_list),), dtype = np.float32)
        subsets_label_list = np.zeros((len(self.pair_list),), dtype = np.int8)
        for index, cur_pair in enumerate(test_pair_list):
            # cur_subset = index // self.num_pair_per_subset
            # cur_id = index % self.num_pair_per_subset
            image_name1 = cur_pair[0]
            image_name2 = cur_pair[1]
            label = cur_pair[2]
            # subsets_label_list[cur_subset][cur_id] = label
            subsets_label_list[index] = label
            feat1 = image_name2feature[image_name1]
            feat2 = image_name2feature[image_name2]
            if not is_normalize:
                feat1 = feat1 / np.linalg.norm(feat1)
                feat2 = feat2 / np.linalg.norm(feat2)
            cur_score = np.dot(feat1, feat2)
            # subsets_score_list[cur_subset][cur_id] = cur_score
            subsets_score_list[index] = cur_score

        # subset_train = np.array([True] * self.num_subset)
        accu_list = []
        # for subset_idx in range(self.num_subset):
        #     test_score_list = subsets_score_list[subset_idx]
        #     test_label_list = subsets_label_list[subset_idx]
        #     #subset_train[subset_idx] = False
        #     train_score_list = subsets_score_list[subset_train].flatten()
        #     train_label_list = subsets_label_list[subset_train].flatten()
        #     subset_train[subset_idx] = True
        #     best_thres = self.getThreshold(train_score_list, train_label_list)
        #     print(best_thres)
        #     #best_thres = 0.3852584743499755
        #     positive_score_list = test_score_list[test_label_list == 1]
        #     negtive_score_list = test_score_list[test_label_list == 0]
        #     true_pos_pairs = np.sum(positive_score_list > best_thres)
        #     true_neg_pairs = np.sum(negtive_score_list < best_thres)
        #     accu_list.append((true_pos_pairs + true_neg_pairs) / self.num_pair_per_subset)

        best_thres = self.getThreshold(subsets_score_list, subsets_label_list)
        # print(best_thres)
        positive_score_list = subsets_score_list[subsets_label_list == 1]
        negtive_score_list = subsets_score_list[subsets_label_list == 0]
        true_pos_pairs = np.sum(positive_score_list > best_thres)
        true_neg_pairs = np.sum(negtive_score_list < best_thres)

        mean = (true_pos_pairs + true_neg_pairs) / len(self.pair_list)
        std = best_thres

        #########
        # mean = np.mean(accu_list)
        # std = np.std(accu_list, ddof=1) / np.sqrt(self.num_subset) #ddof=1, division 9.

        # store predicted result
        # print(self.result_or_not)
        if self.result_or_not == 'record result':
            result_list_file_buf = open('./result/result_' + self.args_mode + '.txt', 'w')
            for index, cur_similarity in enumerate(subsets_score_list):
                if (cur_similarity > best_thres):
                    result_list_file_buf.write('1\n')
                else:
                    result_list_file_buf.write('0\n')

        return mean, std

    def getThreshold(self, score_list, label_list, num_thresholds=1000):
        """Get the best threshold by train_score_list and train_label_list.
        Args:
            score_list(ndarray): the score list of all pairs.
            label_list(ndarray): the label list of all pairs.
            num_thresholds(int): the number of threshold that used to compute roc.
        Returns:
            best_thres(float): the best threshold that computed by train set.
        """
        pos_score_list = score_list[label_list == 1]
        neg_score_list = score_list[label_list == 0]
        pos_pair_nums = pos_score_list.size
        neg_pair_nums = neg_score_list.size
        score_max = np.max(score_list)
        score_min = np.min(score_list)
        score_span = score_max - score_min
        step = score_span / num_thresholds
        threshold_list = score_min +  step * np.array(range(1, num_thresholds + 1)) 
        fpr_list = []
        tpr_list = []
        for threshold in threshold_list:
            fpr = np.sum(neg_score_list > threshold) / neg_pair_nums
            tpr = np.sum(pos_score_list > threshold) /pos_pair_nums
            fpr_list.append(fpr)
            tpr_list.append(tpr)
        fpr = np.array(fpr_list)
        tpr = np.array(tpr_list)
        best_index = np.argmax(tpr-fpr)
        best_thres = threshold_list[best_index]
        return  best_thres
