import numpy as np
from collections import namedtuple


Metrics = namedtuple('Metrics', ('accuracy', 'precision', 'recall'))


def calculate_metrics(predicted: np.ndarray, answer: np.ndarray) -> Metrics:
    assert predicted.shape == answer.shape and predicted.ndim == 2

    total_samples = answer.shape[0]
    predicted_true = (predicted == 1)
    answer_true = (answer == 1)
    true_positives = (predicted_true & answer_true).sum(axis=0)
    true_negatives = (~predicted_true & ~answer_true).sum(axis=0)

    accuracy = (true_positives + true_negatives) / total_samples

    # precision
    predicted_positives = predicted_true.sum(axis=0)
    precision = np.zeros(answer.shape[1])
    ix = predicted_positives > 0
    precision[ix] = true_positives[ix] / predicted_positives[ix]

    # recall
    answer_positives = answer_true.sum(axis=0)
    recall = np.zeros_like(precision)
    ix = answer_positives > 0
    recall = true_positives[ix] / answer_positives[ix]

    return Metrics(accuracy, precision, recall)


def calculate_metrics_mean(predicted: np.ndarray,
                           answer: np.ndarray) -> Metrics:
    assert predicted.shape == answer.shape
    size = 1
    for x in predicted.shape:
        size *= x

    mask_pred = (predicted == 1)
    mask_ans = (answer == 1)
    true_positives = (mask_pred & mask_ans).sum()
    true_negatives = (~mask_pred & ~mask_ans).sum()
    accuracy = (true_positives + true_negatives) / size

    precision = 0.0
    predicted_positives = mask_pred.sum()
    if predicted_positives > 0:
        precision = true_positives / predicted_positives

    recall = 0.0
    actual_positives = mask_ans.sum()
    if actual_positives > 0:
        recall = true_positives / actual_positives

    return Metrics(accuracy, precision, recall)


if __name__ == '__main__':
    # run some tests
    y = np.array([[0, 1], [1, 0], [0, 1]])

    # all zeros
    x = np.zeros_like(y)
    m = calculate_metrics(x, y)
    assert (
        np.allclose(m.accuracy, [2/3, 1/3]) and
        np.allclose(m.precision, [0.0, 0.0]) and
        np.allclose(m.recall, [0.0, 0.0])
    )

    # all ones
    x = np.ones_like(y)
    m = calculate_metrics(x, y)
    assert (
        np.allclose(m.accuracy, [1/3, 2/3]) and
        np.allclose(m.precision, [1/3, 2/3]) and
        np.allclose(m.recall, [1, 1])
    )

    # all accurate
    m = calculate_metrics(y, y)
    assert (
        np.allclose(m.accuracy, [1, 1]) and
        np.allclose(m.precision, [1, 1]) and
        np.allclose(m.recall, [1, 1])
    )
