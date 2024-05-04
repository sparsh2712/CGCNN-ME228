import shutil
import sys
import time
import warnings
from random import sample

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn import metrics
from torch.autograd import Variable


from data import CIFData
from data import collate_pool, get_train_val_test_loader
from cgcnn import CrystalGraphConvNet


def main():
    global best_mae_error  # Declare global variable best_mae_error
    
    # Initialize CIFData object with paths to CIF files, JSON file, and property CSV file
    dataset = CIFData(cif_directory='/Users/sparsh/Desktop/College core/AI_DS_228/CGCNN-ME228/data/cif_files',
                      json_path='/Users/sparsh/Desktop/College core/AI_DS_228/CGCNN-ME228/data/atom_init.json',
                      property_id_path='/Users/sparsh/Desktop/College core/AI_DS_228/CGCNN-ME228/data/id_prop.csv')
    
    # Define collate function for data loader
    collate_fn = collate_pool
    
    # Split dataset into train, validation, and test loaders
    train_loader, val_loader, test_loader = get_train_val_test_loader(
        dataset=dataset,
        collate_fn=collate_fn,
        batch_size=256,
        train_ratio=0.6,
        num_workers=1,
        val_ratio=0.2,
        test_ratio=0.2,
        return_test=True)

    # Obtain target value normalizer
    if len(dataset) < 500:
        warnings.warn('Dataset has less than 500 data points. '
                    'Lower accuracy is expected. ')
        # If the dataset is small, use all data points for normalization
        sample_data_list = [dataset[i] for i in range(len(dataset))]
    else:
        # If the dataset is large, randomly sample 500 data points for normalization
        sample_data_list = [dataset[i] for i in sample(range(len(dataset)), 500)]
    _, sample_target, _ = collate_pool(sample_data_list)
    normalizer = Normalizer(sample_target)

    # Build model
    structures, _, _ = dataset[0]
    orig_atom_fea_len = structures[0].shape[-1]
    nbr_fea_len = structures[1].shape[-1]
    model = CrystalGraphConvNet(orig_atom_fea_len, nbr_fea_len,
                                atom_fea_len=103,
                                n_conv=3,
                                h_fea_len=128,
                                n_h=1,
                                )

    model.cuda()  # Move model to GPU
    
    optimizer = optim.Adam(model.parameters(), 0.01, weight_decay=None)  # Initialize optimizer
    
    criterion = nn.MSELoss()  # Define loss function
    
    # Train the model for multiple epochs
    for epoch in range(30):
        # Train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, normalizer)

        # Evaluate on validation set
        mae_error = validate(val_loader, model, criterion, normalizer)

        # Remember the best mae_error and save checkpoint
        is_best = mae_error < best_mae_error
        best_mae_error = min(mae_error, best_mae_error)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_mae_error': best_mae_error,
            'optimizer': optimizer.state_dict(),
            'normalizer': normalizer.state_dict(),
        }, is_best)

    # Test the best model
    print('---------Evaluate Model on Test Set---------------')
    best_checkpoint = torch.load('model_best.pth.tar')  # Load best model checkpoint
    model.load_state_dict(best_checkpoint['state_dict'])  # Load best model weights
    validate(test_loader, model, criterion, normalizer, test=True)  # Evaluate model on test set



def train(train_loader, model, criterion, optimizer, epoch, normalizer):
    # Initialize meters to keep track of time, loss, and error
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    mae_errors = AverageMeter()

    # switch to train mode
    model.train()  # Set the model to train mode (enable dropout, batch normalization, etc.)

    # Iterate over batches in the train loader
    for i, (input, target, _) in enumerate(train_loader):

        # Measure data loading time
        input_var = (Variable(input[0].cuda(non_blocking=True)),  # Convert input data to CUDA variables
                        Variable(input[1].cuda(non_blocking=True)),
                        input[2].cuda(non_blocking=True),
                        [crys_idx.cuda(non_blocking=True) for crys_idx in input[3]])
        
        # Normalize target values using the provided normalizer
        target_normed = normalizer.norm(target)
        target_var = Variable(target_normed.cuda(non_blocking=True))  # Convert target data to CUDA variable

        # Forward pass: compute model output
        output = model(*input_var)
        loss = criterion(output, target_var)  # Compute loss using the provided criterion

        # Compute mean absolute error (MAE) between denormalized output and target
        mae_error = mae(normalizer.denorm(output.data.cpu()), target)

        # Update loss and error meters
        losses.update(loss.data.cpu(), target.size(0))
        mae_errors.update(mae_error, target.size(0))

        # Backpropagation: compute gradient and update model parameters
        optimizer.zero_grad()  # Clear accumulated gradients
        loss.backward()  # Backpropagate the loss
        optimizer.step()  # Update model parameters using the computed gradients


def validate(val_loader, model, criterion, normalizer):
    # Initialize meters to keep track of time, loss, and error
    batch_time = AverageMeter()
    losses = AverageMeter()
    mae_errors = AverageMeter()

    # switch to evaluate mode
    model.eval()  # Set the model to evaluation mode (disable dropout, batch normalization, etc.)

    # Iterate over batches in the validation loader
    for i, (input, target, batch_cif_ids) in enumerate(val_loader):
        with torch.no_grad():  # Context manager to disable gradient computation
            # Convert input data to CUDA variables
            input_var = (Variable(input[0].cuda(non_blocking=True)),
                            Variable(input[1].cuda(non_blocking=True)),
                            input[2].cuda(non_blocking=True),
                            [crys_idx.cuda(non_blocking=True) for crys_idx in input[3]])

        # Normalize target values using the provided normalizer
        target_normed = normalizer.norm(target)

        with torch.no_grad():  # Context manager to disable gradient computation
            # Convert target data to CUDA variable
            target_var = Variable(target_normed.cuda(non_blocking=True))

        # Forward pass: compute model output
        output = model(*input_var)
        
        # Compute loss using the provided criterion
        loss = criterion(output, target_var)
        
        # Compute mean absolute error (MAE) between denormalized output and target
        mae_error = mae(normalizer.denorm(output.data.cpu()), target)
        
        # Update loss and error meters
        losses.update(loss.data.cpu().item(), target.size(0))
        mae_errors.update(mae_error, target.size(0))

    return mae_errors.avg  # Return the average mean absolute error over all batches

    

class Normalizer(object):
    """Normalize a Tensor and restore it later. """

    def __init__(self, tensor):
        """Initialize the Normalizer object with mean and standard deviation of the input tensor."""
        self.mean = torch.mean(tensor)  # Calculate mean of the input tensor
        self.std = torch.std(tensor)    # Calculate standard deviation of the input tensor

    def norm(self, tensor):
        """Normalize the input tensor using the previously calculated mean and standard deviation."""
        return (tensor - self.mean) / self.std  # Normalize the input tensor

    def denorm(self, normed_tensor):
        """Denormalize the normalized tensor using the previously calculated mean and standard deviation."""
        return normed_tensor * self.std + self.mean  # Denormalize the normalized tensor

    def state_dict(self):
        """Return a dictionary containing the state of the normalizer."""
        return {'mean': self.mean,  # Return mean and standard deviation as the state dictionary
                'std': self.std}

    def load_state_dict(self, state_dict):
        """Load the state dictionary to restore the mean and standard deviation of the normalizer."""
        self.mean = state_dict['mean']  # Load mean from the state dictionary
        self.std = state_dict['std']    # Load standard deviation from the state dictionary



def mae(prediction, target):
    """
    Computes the mean absolute error between prediction and target

    Parameters
    ----------
    prediction: torch.Tensor (N, 1)
        Tensor containing the predicted values
    target: torch.Tensor (N, 1)
        Tensor containing the target values
    Returns
    -------
    torch.Tensor
        Mean absolute error between prediction and target
    """
    return torch.mean(torch.abs(target - prediction))  # Calculate mean absolute error


def class_eval(prediction, target):
    """
    Evaluate classification performance metrics.

    Parameters:
    -----------
    prediction : torch.Tensor
        Predicted values
    target : torch.Tensor
        True labels

    Returns:
    --------
    accuracy : float
        Accuracy of the classification
    precision : float
        Precision of the classification
    recall : float
        Recall of the classification
    fscore : float
        F1 score of the classification
    auc_score : float
        Area under the ROC curve score of the classification
    """
    prediction = np.exp(prediction.numpy())  # Exponentiate the prediction to get probabilities
    target = target.numpy()  # Convert target tensor to numpy array
    pred_label = np.argmax(prediction, axis=1)  # Predicted labels
    target_label = np.squeeze(target)  # True labels
    if not target_label.shape:
        target_label = np.asarray([target_label])  # Convert scalar target to array if necessary
    if prediction.shape[1] == 2:  # Check if binary classification
        # Compute precision, recall, F1 score, and accuracy
        precision, recall, fscore, _ = metrics.precision_recall_fscore_support(
            target_label, pred_label, average='binary')
        # Compute area under the ROC curve
        auc_score = metrics.roc_auc_score(target_label, prediction[:, 1])
        # Compute accuracy
        accuracy = metrics.accuracy_score(target_label, pred_label)
    else:
        raise NotImplementedError("Multiclass evaluation is not implemented yet.")
    return accuracy, precision, recall, fscore, auc_score  # Return evaluation metrics


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """
    Save model checkpoint to file.

    Parameters:
    -----------
    state : dict
        Dictionary containing model state, optimizer state, and other information.
    is_best : bool
        Flag indicating whether the current checkpoint is the best one so far.
    filename : str, optional (default='checkpoint.pth.tar')
        Name of the checkpoint file to save.
    """
    torch.save(state, filename)  # Save model checkpoint to specified file
    if is_best:  # If the current checkpoint is the best one
        # Copy the checkpoint file to a file named 'model_best.pth.tar'
        shutil.copyfile(filename, 'model_best.pth.tar')



def adjust_learning_rate(optimizer, epoch, k):
    """
    Sets the learning rate to the initial LR decayed by 10 every k epochs.

    Parameters:
    -----------
    optimizer : torch.optim.Optimizer
        The optimizer for which the learning rate will be adjusted.
    epoch : int
        The current epoch number.
    k : int
        Number of epochs after which the learning rate will be decayed by a factor of 10.
    """
    assert type(k) is int  # Ensure k is an integer

    # Compute the new learning rate using exponential decay
    lr = 0.01 * (0.1 ** (epoch // k))

    # Update the learning rate for all parameter groups in the optimizer
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr  # Set the learning rate for the current parameter group


if __name__ == '__main__':
    main()
