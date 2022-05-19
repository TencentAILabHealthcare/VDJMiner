# -*- coding: utf-8 -*-
"""
This script is used for evaluation using optimal cutpoints
"""

import os
import pandas as pd
import numpy as np
import bootstrap as boot
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve


# helper functions
def get_optimized_point_maxSensitivitySpecificity(y, scores, pos_label=1):
    fpr, tpr, thresholds = roc_curve(y, scores, pos_label=pos_label)
    gmeans = np.sqrt(tpr * (1 - fpr))
    optimal_idx = np.argmax(gmeans)
    optimal_threshold = thresholds[optimal_idx]
    return optimal_threshold


def get_prediction_from_probability(input, threshold):
    output = [item >= threshold for item in input]
    return output


def auc_calculate(y_test, y_pred_proba):
    auc = roc_auc_score(y_test, y_pred_proba)
    return auc


def performance_metric(ground_truth,
                       predicted_result,
                       output_item,
                       use_weight,
                       verbose,
                       eplison=1e-6):
    """
    calculate the TP, FP, TN, FN
    """

    per_dict = dict()
    conf_matrix = confusion_matrix(ground_truth, predicted_result)
    FP = conf_matrix.sum(axis=0) - np.diag(conf_matrix)
    FN = conf_matrix.sum(axis=1) - np.diag(conf_matrix)
    TP = np.diag(conf_matrix)
    TN = conf_matrix.sum() - (FP + FN + TP)

    # Sensitivity, hit rate, recall, or true positive rate
    TPR = TP / (TP + FN + eplison)
    # Specificity or true negative rate
    TNR = TN / (TN + FP + eplison)
    # Precision or positive predictive value
    PPV = TP / (TP + FP + eplison)
    # Negative predictive value
    NPV = TN / (TN + FN + eplison)
    # Fall out or false positive rate
    FPR = FP / (FP + TN + eplison)
    # False negative rate
    FNR = FN / (TP + FN + eplison)
    # False discovery rate
    FDR = FP / (TP + FP + eplison)

    # Overall accuracy
    ACC = (TP + TN) / (TP + FP + FN + TN + eplison)

    # F1 score = 2 * (precision * recall) / (precision + recall)
    F1 = 2 * (TPR * PPV) / (PPV + TPR + eplison)

    if (TP.shape[0] == 2) and (FN.shape[0] == 2):
        per_dict['FP'] = FP[1]
        per_dict['FN'] = FN[1]
        per_dict['TP'] = TP[1]
        per_dict['TN'] = TN[1]

        if use_weight:
            weights = [TP[0] + FN[0], TP[1] + FN[1]]  # weights is true class 0 and true class 1

            weights = np.array(weights)
            TPR = np.average(TPR, weights=weights)
            TNR = np.average(TNR, weights=weights)
            PPV = np.average(PPV, weights=weights)
            NPV = np.average(NPV, weights=weights)
            FPR = np.average(FPR, weights=weights)
            FNR = np.average(FNR, weights=weights)
            FDR = np.average(FDR, weights=weights)
            ACC = np.average(ACC, weights=weights)
            F1 = np.average(F1, weights=weights)
            per_dict['TPR'] = TPR
            per_dict['TNR'] = TNR
            per_dict['PPV'] = PPV
            per_dict['NPV'] = NPV
            per_dict['FPR'] = FPR
            per_dict['FNR'] = FNR
            per_dict['FDR'] = FDR
            per_dict['ACC'] = ACC
            per_dict['F1'] = F1
        else:
            per_dict['TPR'] = TPR[1]
            per_dict['TNR'] = TNR[1]
            per_dict['PPV'] = PPV[1]
            per_dict['NPV'] = NPV[1]
            per_dict['FPR'] = FPR[1]
            per_dict['FNR'] = FNR[1]
            per_dict['FDR'] = FDR[1]
            per_dict['ACC'] = ACC[1]
            per_dict['F1'] = F1[1]
    else:
        return None

    if verbose:
        print('Sensitivity/recall:' + str(TPR))
        print('Specificity       :' + str(TNR))
        print('Precision/PPV     :' + str(PPV))
        print('NPV               :' + str(NPV))

    if output_item == 'ALL':
        return per_dict
    else:
        return per_dict[output_item]


def calculate_ci_bootstraping(y_test,
                              y_predict,
                              y_possibility,
                              alpha=0.05,
                              iter_num=10000,
                              method='bca',
                              use_weight=False,
                              multi_level=True,
                              metric_list=('AUC', 'TPR', 'TNR', 'PPV', 'NPV', 'FDR', 'ACC', 'F1')
                              ):
    """
    when using the multi_level, we should make sure that the fist item in the data tuple 
        should be the GoundTruth classification.
    multilevel: makesure each class has the same samplers as itself druing sampling

    alpha: float or iterable, optional
    The percentiles to use for the confidence interval (default=0.05). If this
    is a float, the returned values are (alpha/2, 1-alpha/2) percentile confidence
    intervals. If it is an iterable, alpha is assumed to be an iterable of
    each desired percentile.

    """
    ci_dict = dict()

    if not isinstance(y_test, (np.ndarray, np.generic)):
        y_test = np.array(y_test)
    if not isinstance(y_predict, (np.ndarray, np.generic)):
        y_predict = np.array(y_predict)

    for metricItem in metric_list:

        if metricItem == 'AUC':
            statfunction = (lambda y_test, y_possbility: auc_calculate(y_test, y_possbility))
            out_low_high = boot.ci(data=(y_test, y_possibility),
                                   statfunction=statfunction,
                                   alpha=alpha,
                                   n_samples=iter_num,
                                   method=method,  # 'pi','bca','abc'
                                   output='lowhigh',
                                   epsilon=1e-6,
                                   multi=True,
                                   multiLevel=multi_level)
        else:
            statfunction = (lambda y_test, y_predict: performance_metric(y_test, y_predict, metricItem, use_weight,
                                                                         verbose=False))
            out_low_high = boot.ci(data=(y_test, y_predict),
                                   statfunction=statfunction,
                                   alpha=alpha,
                                   n_samples=iter_num,
                                   method=method,  # 'pi','bca','abc'
                                   output='lowhigh',
                                   epsilon=1e-6,
                                   multi=True,
                                   multiLevel=multi_level)

        ci_dict['{}_low'.format(metricItem)] = out_low_high[0]
        ci_dict['{}_high'.format(metricItem)] = out_low_high[1]
    return ci_dict


def data_reader(input_file, subject_id_title, ground_truth_title, prediction_title):
    df = pd.read_csv(input_file)
    if subject_id_title:
        subject_id = df[subject_id_title].to_numpy()
    else:
        subject_id = None
    ground_truth = df[ground_truth_title].to_numpy()
    prediction = df[prediction_title].to_numpy()
    return subject_id, ground_truth, prediction


def evaluate(y_test,
             y_predict,
             y_predict_probability,
             class_note,
             output_file,
             bootstrap_method='bca',
             bootstrap_alpha=0.05,
             bootstrap_iter_num=2000,
             use_weight=False,
             multi_level=True,
             seed=64
             ):
    np.random.seed(seed)
    per_dict = performance_metric(y_test, y_predict, 'ALL', use_weight, verbose=False)
    per_dict['AUC'] = auc_calculate(y_test, y_predict_probability)
    ci_dict = calculate_ci_bootstraping(y_test,
                                        y_predict,
                                        y_predict_probability,
                                        alpha=bootstrap_alpha,
                                        iter_num=bootstrap_iter_num,
                                        method=bootstrap_method,
                                        use_weight=use_weight,
                                        multi_level=multi_level)

    with open(output_file, mode='a') as file_handle:
        file_handle.write('------------------------------------------------------------------\n')
        file_handle.write('{}           Value        Lower Limit     Upper Limit     \n'.format(class_note))
        file_handle.write('------------------------------------------------------------------\n')
        file_handle.write(
            'AUC      :           {:.3f}           {:.3f}           {:.3f}        \n'.format(per_dict['AUC'],
                                                                                             ci_dict['AUC_low'],
                                                                                             ci_dict['AUC_high']))
        file_handle.write(
            'Accuracy :           {:.3f}           {:.3f}           {:.3f}        \n'.format(per_dict['ACC'],
                                                                                             ci_dict['ACC_low'],
                                                                                             ci_dict['ACC_high']))
        file_handle.write(
            'F1-Score :           {:.3f}           {:.3f}           {:.3f}        \n'.format(per_dict['F1'],
                                                                                             ci_dict['F1_low'],
                                                                                             ci_dict['F1_high']))
        file_handle.write('------------------------------------------------------------------\n')


def eval_result(test_possibility, test_label, result_dir, target, use_weight=False):
    """ evaluate performance of model predictions

    Args:
        test_possibility: possibility from model.predict_proba(), [n_sample, n_classes]
        test_label: test data label, [n_sample]
        result_dir: data directory to store results
        target: target name, eg. Cancer
        use_weight: True/False whether using weighted average 

    Returns:

    """
    # calculate class number
    os.makedirs(result_dir, exist_ok=True)
    class_num = test_possibility.shape[1]
    output_file = os.path.join(result_dir, target + '.txt')
    # Delete previous file
    if os.path.exists(output_file):
        os.remove(output_file)
    if class_num > 2:
        test_label_onehot = OneHotEncoder(max_categories=class_num, sparse=False).fit_transform(test_label)
        for i in range(class_num):
            ground_truth = test_label_onehot[:, i]
            prediction = test_possibility[:, i]
            class_note = 'Class:' + str(i).ljust(4)
            cutpoint_value = get_optimized_point_maxSensitivitySpecificity(y=ground_truth, scores=prediction)
            prediction_label = get_prediction_from_probability(prediction, cutpoint_value)
            evaluate(y_test=ground_truth,
                     y_predict=prediction_label,
                     y_predict_probability=prediction,
                     class_note=class_note,
                     output_file=output_file,
                     use_weight=use_weight
                     )
    else:
        ground_truth = test_label
        prediction = test_possibility[:, 1]
        class_note = ' ' * 10
        cutpoint_value = get_optimized_point_maxSensitivitySpecificity(y=ground_truth, scores=prediction)
        prediction_label = get_prediction_from_probability(prediction, cutpoint_value)
        evaluate(y_test=ground_truth,
                 y_predict=prediction_label,
                 y_predict_probability=prediction,
                 class_note=class_note,
                 output_file=output_file,
                 use_weight=use_weight
                 )
