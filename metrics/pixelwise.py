import torch


def calculate_metrics(tp, tn, fp, fn, use_epsilon=True, epsilon=1e-9):
    """
    Calculate accuracy, precision, recall, and F1 score.

    Parameters:
    tp (int): True positives
    tn (int): True negatives
    fp (int): False positives
    fn (int): False negatives
    use_epsilon (bool): Whether to use epsilon to avoid division by zero (default: False)
    epsilon (float): Small value to add to denominators to avoid division by zero (default: 1e-9)

    Returns:
    dict: A dictionary containing accuracy, precision, recall, and F1 score.
    """

    if use_epsilon:
        accuracy = (tp + tn) / (tp + tn + fp + fn + epsilon)
        precision = tp / (tp + fp + epsilon)
        recall = tp / (tp + fn + epsilon)
        f1 = 2 * precision * recall / (precision + recall + epsilon)
    else:
        accuracy = (tp + tn) / (tp + tn + fp + fn) if tp + tn + fp + fn != 0 else 0
        precision = tp / (tp + fp) if tp + fp != 0 else 0
        recall = tp / (tp + fn) if tp + fn != 0 else 0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall != 0
            else 0
        )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
    }


def calculate_confusion_matrix(y_logits, y, threshold=0.5):
    """
    Calculate true positives, true negatives, false positives, and false negatives.

    Parameters:
     y_logits (torch.Tensor): The raw output logits from the model (before sigmoid).
    y (torch.Tensor): The ground truth labels.
    threshold (float): The threshold to classify predictions as positive (default: 0.5).

    Returns:
    dict: A dictionary containing tp (true positives), tn (true negatives),
          fp (false positives), and fn (false negatives).
    """

    # Apply sigmoid and threshold to convert logits to binary predictions
    y_preds = torch.sigmoid(y_logits.detach()).view(-1) >= threshold
    y_true = y.detach().view(-1) >= threshold

    # Calculate confusion matrix components
    tp = (y_preds & y_true).sum().item()
    tn = ((~y_preds) & (~y_true)).sum().item()
    fp = (y_preds & (~y_true)).sum().item()
    fn = ((~y_preds) & y_true).sum().item()

    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}
