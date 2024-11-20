import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment


class HungarianMatcher(nn.Module):
    def __init__(self, distance_weight=1.0, logits_weight=1.0):
        """
        Initializes the HungarianMatcher module.

        Args:
            missing_cost (float, optional): Cost for each missing predicted point. Should be specified in runtime.
        """
        super(HungarianMatcher, self).__init__()
        self.distance_weight = distance_weight
        self.logits_weight = logits_weight

    def forward(self, coordinates_gt, coordinates_pred, logits_gt, logits_pred):
        """
        Computes the optimal assignment indices and total cost for matching ground truth (GT) coordinates
        and logits with predicted coordinates and logits.

        Args:
            coordinates_gt (torch.Tensor): Ground truth coordinates with shape (N, D), where N is the number of
                                           GT points and D is the dimensionality.
            coordinates_pred (torch.Tensor): Predicted coordinates with shape (M, D), where M is the number of
                                             predicted points and D is the dimensionality.
            logits_gt (torch.Tensor): Ground truth logits with shape (N, L), where L is the number of classes.
            logits_pred (torch.Tensor): Predicted logits with shape (M, L), where L is the number of classes.

        Returns:
            indices (tuple of numpy.ndarray): A tuple of two arrays containing the indices of the optimal assignment
                                              for ground truth and predicted points.
            cost (float): The total cost, including both the missing cost and the distance cost.

        Notes:
            - The function calls the `match` method to compute the cost matrix and optimal assignment indices.
            - The total cost includes the distance cost for the assigned pairs.
        """

        indices, cost_distance = self.match(
            coordinates_gt, coordinates_pred, logits_gt, logits_pred
        )

        # Add the distance cost for the assigned pairs to the total cost
        cost = cost_distance[indices[0], indices[1]].sum()

        return indices, cost

    def match(self, coordinates_gt, coordinates_pred, logits_gt, logits_pred):
        """
        Computes the cost matrix and optimal assignment indices for matching ground truth (GT) coordinates
        and logits with predicted coordinates and logits.

        Args:
            coordinates_gt (torch.Tensor): Ground truth coordinates with shape (N, D), where N is the number of
                                           GT points and D is the dimensionality.
            coordinates_pred (torch.Tensor): Predicted coordinates with shape (M, D), where M is the number of
                                             predicted points and D is the dimensionality.
            logits_gt (torch.Tensor): Ground truth logits with shape (N, L), where L is the number of classes.
            logits_pred (torch.Tensor): Predicted logits with shape (M, L), where L is the number of classes.

        Returns:
            indices (tuple of numpy.ndarray): A tuple of two arrays containing the indices of the optimal assignment
                                              for ground truth and predicted points.
            cost_distance (torch.Tensor): The distance cost matrix.
            cost_matrix (torch.Tensor): The combined cost matrix including both distance and logits costs.

        Notes:
            - The function computes a cost matrix based on the Euclidean distances between the coordinates and the
              logits of the ground truth and predicted points.
            - The linear_sum_assignment function from scipy.optimize is used to find the optimal assignment that
              minimizes the total cost.
            - The total cost includes the cost for any missing predicted points and the distance cost for the
              assigned pairs.
        """

        # Calculate the distance cost matrix
        cost_distance = torch.cdist(coordinates_gt.float(), coordinates_pred.float())

        cost_matrix = cost_distance * self.distance_weight

        if self.logits_weight > 0:
            # Calculate the logits cost matrix
            cost_logits = torch.cdist(logits_gt, logits_pred, p=1)

            # Combine distance and logits cost matrices
            cost_matrix += cost_logits * self.logits_weight

        # Get the optimal assignment indices using the linear sum assignment method
        indices = linear_sum_assignment(cost_matrix)

        return indices, cost_distance
