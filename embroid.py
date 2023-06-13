import warnings
warnings.filterwarnings("ignore") # Suppress warnings from FlyingSquid

import numpy as np
from flyingsquid.label_model import LabelModel



def run_embroid(votes, nn_info, knn=10, thresholds=[[0.5, 0.5]]):
    """
    Implements Embroid.

    Parameters
    ----------

    votes : ndarray of shape (n_samples, n_sources)
        Predictions of LMs. Should be in 1/-1 space.

    nn_info: ndarray of shape (n_embeddings, n_samples, d)
        Nearest neighbor information for each of the n_embedding spaces.
        nn_info[i, t, l] is the index of the lth nearest-neighbor in ith
        embedding space for sample t.

    knn: int
        Number of neighbors to use when computing neighborhood votes.

    thresholds: ndarray of shape (n_embeddings, 2)
        The tau threshold used for computing majority votes.
    """

    # Check that votes are in 1/-1 space
    assert sorted(np.unique(votes)) in [[1], [-1], [-1, 1]], np.unique(votes)
    n_samples, n_sources = votes.shape
    n_embeddings = len(nn_info)

    # compute neighborhood votes for each source
    inputs = [votes]
    for i in range(n_embeddings):
        S = np.zeros((n_samples, n_sources))
        for j in range(n_sources):
            # Convert prediction of source j to index space (0, 1)
            j_prediction = (votes[:, j] + 1) / 2

            # Compute fraction of nearest neighbor votes for positive class
            neighbor_pos_frac = j_prediction[nn_info[i, :, 1 : 1 + knn]].mean(axis=1)

            # Construct neighborhood votes from fractions
            shrunk_neighbor_votes = np.zeros(len(neighbor_pos_frac))
            idxs = np.where(neighbor_pos_frac >= thresholds[j][1])
            shrunk_neighbor_votes[idxs] = 1
            idxs = np.where((1 - neighbor_pos_frac) >= thresholds[j][0])
            shrunk_neighbor_votes[idxs] = -1
            S[:, j] = shrunk_neighbor_votes
        inputs.append(S)

    # Stack votes and S
    mod_votes = np.concatenate(inputs, axis=1)
    assert mod_votes.shape[1] == n_sources * (len(inputs))

    label_model = LabelModel(n_sources * (len(inputs)))
    label_model.fit(mod_votes)
    preds = label_model.predict(mod_votes).ravel()
    return preds
