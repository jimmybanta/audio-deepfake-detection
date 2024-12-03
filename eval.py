''' Functions for evaluating models. '''

import torch
import numpy as np
from sklearn import metrics

from data.import_data import get_dataloader




def get_optimalROC(probs, targets):
    '''
    Returns the optimal threshold for a model, given the predicted probabilities and the targets.

    Parameters 
    ----------
    probs : torch.tensor
        The predicted probabilities of the model
    targets : torch.tensor
        The true labels of the data

    Returns
    -------
    best_thresh : float
        The optimal threshold for the model
    '''

    probs = probs.cpu().numpy()
    targets = targets.cpu().numpy()

    fpr, tpr, thresholds = metrics.roc_curve(targets, probs)

    J = tpr - fpr

    ix = np.argmax(J)
    best_thresh = thresholds[ix]

    return best_thresh
def eval_model(model, dataset):
    '''
    Given a model and a dataset, evaluate the model on the dataset.add

    Parameters
    ----------
    model : Model
        The model to evaluate
    dataset : TensorDataset
        The dataset to evaluate the model on

    '''

    # get the dataloader, all in one batch
    loader = get_dataloader(dataset, batch_size=100000)

    # set the model to evaluation
    model.eval()

    with torch.no_grad():

        for inputs, targets in loader:
            probs = model(inputs)

            probs = probs[:, 0]
            targets = targets[:, 0]

            thresh = get_optimalROC(probs, targets)
            print(f'Optimal Threshold: {thresh}')
            preds = (probs >= thresh).long()

            probs = probs.cpu().numpy()
            preds = preds.cpu().numpy()
            targets = targets.cpu().numpy()

            roc = metrics.roc_auc_score(targets, probs)

            tn, fp, fn, tp = metrics.confusion_matrix(targets, preds).ravel()

    
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    sens = tp / (tp + fn)
    spec = tn / (tn + fp)
    ppv = tp / (tp + fp)
    npv = tn / (tn + fn)

    print(f'ROC: {roc}')
    print(f'Accuracy: {accuracy}')
    print(f'Sensitivity: {sens}')
    print(f'Specificity: {spec}')
    print(f'PPV: {ppv}')
    print(f'NPV: {npv}')

    metrics.ConfusionMatrixDisplay(metrics.confusion_matrix(targets, preds)).plot()
    