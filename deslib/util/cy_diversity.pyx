# cimport numpy as np

# DTYPE = np.float64

cdef float _process_predictions_agreement(y, y_pred1, y_pred2):
    """
	Cython implementation of the same function in `utils/diversity`.
    note: changed to return only agreement.


    Pre-process the predictions of a pair of base classifiers for the
    computation of the diversity measures

    Parameters
    ----------
    y : array of shape (n_samples):
        class labels of each sample.

    y_pred1 : array of shape (n_samples):
              predicted class labels by the classifier 1 for each sample.

    y_pred2 : array of shape (n_samples):
              predicted class labels by the classifier 2 for each sample.

    Returns
    -------
    N00 : Percentage of samples that both classifiers predict the wrong label

    N10 : Percentage of samples that only classifier 2 predicts the wrong label

    N10 : Percentage of samples that only classifier 1 predicts the wrong label

    N11 : Percentage of samples that both classifiers predict the correct label
    """

    cdef int size_y = len(y)
    cdef int size_y_pred1 = len(y_pred1)
    cdef int size_y_pred2 = len(y_pred2)
    if size_y != size_y_pred1 or size_y != size_y_pred2:
        raise ValueError(
            'The vector with class labels must have the same size.')

    cdef:
        float N00 = 0.0, N10 = 0.0, N01 = 0.0, N11 = 0.0
        int index
    for index in range(size_y):
        if y_pred1[index] == y[index] and y_pred2[index] == y[index]:
            N11 += 1.0
        elif y_pred1[index] != y[index] and y_pred2[index] != y[index]:
            N00 += 1.0


    return (N00 + N11) / size_y


def test_predictions():
    import numpy as np
    y_pred_classifier1 = np.array([0, 0, 0, 1, 0, 1, 0, 0, 0, 0])
    y_pred_classifier2 = np.array([1, 0, 0, 1, 1, 0, 0, 0, 1, 1])
    y_real = np.array([0, 0, 1, 0, 0, 0, 0, 1, 1, 1])
    print(f"the agreement meaasure is:")
    print(_process_predictions_agreement(y_real,
                                         y_pred_classifier1,
                                         y_pred_classifier2))
