import os
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report


def eval_result(test_possibility, test_label, result_dir, target):
    """ evaluate performance of model predictions

    Args:
        test_possibility: possibility from model.predict_proba(), [n_sample, n_classes]
        test_label: test data label, [n_sample]
        result_dir: data directory to store results
        target: target name, eg. Cancer

    Returns:

    """
    os.makedirs(result_dir, exist_ok=True)
    nb_classes = test_possibility.shape[1]
    Y_test = to_categorical(test_label, nb_classes)
    Y_predict = np.argmax(test_possibility, axis=1)

    label_df = pd.DataFrame(Y_test, columns=[f'label_is_{i}' for i in range(Y_test.shape[1])])
    pred_df = pd.DataFrame(test_possibility, columns=[f'possibility_label_{i}' for i in range(nb_classes)])
    result_df = pd.concat([label_df, pred_df.reindex(label_df.index)], axis=1)
    result_df['ground_truth'] = test_label
    result_df['prediction'] = Y_predict

    output_file = os.path.join(result_dir, f'{target}_Prediction_result.csv')
    result_df.to_csv(output_file, index=False)

    # classification report
    class_report = classification_report(test_label, Y_predict, output_dict=False, digits=4)
    with open(os.path.join(result_dir, f'{target}.txt'), 'w') as f:
        f.write(class_report)


def to_categorical(y, num_classes=None, dtype='float32'):
    """
    Converts a class vector (integers) to binary class matrix.
    """
    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype=dtype)
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical
