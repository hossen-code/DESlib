cimport numpy as np

# DTYPE = np.float64

cdef _process_predictions(y, y_pred1, y_pred2):
    """
	Cython implementation of the same function in `utils/diversity`.


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
        elif y_pred1[index] == y[index] and y_pred2[index] != y[index]:
            N10 += 1.0
        elif y_pred1[index] != y[index] and y_pred2[index] == y[index]:
            N01 += 1.0
        else:
            N00 += 1.0


    return N00 / size_y, N10 / size_y, N01 / size_y, N11 / size_y


def agreement_measure(y, y_pred1, y_pred2):
    """
    Cython implementation of the same function in `utils/diversity`.

    Calculates the agreement measure between a pair of classifiers. This
    measure is calculated by the frequency that both classifiers either
    obtained the correct or incorrect prediction for any given sample

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
    agreement : The frequency at which both classifiers agrees
    """
    N00, _, _, N11 = _process_predictions(y, y_pred1, y_pred2)
    cdef float agreement = N00 + N11
    return agreement