import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def evaluate_metrics(
    model: torch.nn.Module,
    val_loader: torch.utils.data.DataLoader,
    loss_function: torch.nn.Module,
    device: torch.device,
):
    """
    Evaluates the performance of a trained model on a validation dataset, calculating loss and common classification metrics.

    Args:
        model (torch.nn.Module): The trained model to evaluate.
        val_loader (torch.utils.data.DataLoader): DataLoader for the validation dataset.
        loss_function (torch.nn.Module): The loss function used for evaluation.
        device (torch.device): The device (CPU or GPU) to use for computation.

    Returns:
        tuple: A tuple containing the validation loss, accuracy, precision, recall, and F1-score.
            - val_loss (float): The average validation loss.
            - acc (float): The accuracy of the model's predictions.
            - precision (float): The macro-averaged precision of the model's predictions.
            - recall (float): The macro-averaged recall of the model's predictions.
            - f1 (float): The macro-averaged F1-score of the model's predictions.
    """

    model.eval()
    val_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for val_inputs, val_targets in val_loader:
            val_inputs, val_targets = val_inputs.to(device), val_targets.to(device)
            val_outputs = model(val_inputs)
            val_loss += loss_function(val_outputs, val_targets).item()

            preds = torch.argmax(val_outputs, dim=1)  # Get class predictions
            all_preds.extend(preds.cpu().numpy())  # Store predictions
            all_targets.extend(val_targets.cpu().numpy())  # Store actual labels

    val_loss /= len(val_loader)

    acc = accuracy_score(all_targets, all_preds)
    precision = precision_score(
        all_targets, all_preds, average="macro", zero_division=0
    )
    recall = recall_score(all_targets, all_preds, average="macro", zero_division=0)
    f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)

    return val_loss, acc, precision, recall, f1
