from sklearn.metrics import normalized_mutual_info_score
from forward_greedy_facility import ForwardGreedyFacility

def evaluate_clustering(y_gt, y_assignment):
    return normalized_mutual_info_score(y_gt, y_assignment)

def run_loss_aug_clustering_on(data, gt_labels, loss_mult, n_clusters=10):
    """

    Parameters
    ----------
    data: NxM array, N:number of points, M embedding size


    Returns
    -------

    """
    clustering_result = ForwardGreedyFacility(n_clusters=n_clusters).loss_augmented_fit(data, gt_labels, loss_mult)
    return clustering_result.labels_

def run_clustering_on(data, n_clusters=10):
    """

    Parameters
    ----------
    data: NxM array, N:number of points, M embedding size


    Returns
    -------

    """
    clustering_result = ForwardGreedyFacility(n_clusters=n_clusters).fit(data)
    return clustering_result.labels_
