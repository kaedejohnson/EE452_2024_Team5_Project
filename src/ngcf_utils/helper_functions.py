'''
Pytorch Implementation of Neural Graph Collaborative Filtering (NGCF) (https://doi.org/10.1145/3331184.3331267)

This file contains the following helper functions:
- early_stopping()
- train()
- split_matrix()
- ndcg_k()
- eval_model

authors: Mohammed Yusuf Noor, Muhammed Imran Özyar, Calin Vasile Simon
'''

import numpy as np
import torch

def early_stopping(log_value, best_value, stopping_step, flag_step, expected_order='asc'):
    """
    Check if early_stopping is needed
    Function copied from original code
    """
    assert expected_order in ['asc', 'des']
    if (expected_order == 'asc' and log_value >= best_value) or (expected_order == 'des' and log_value <= best_value):
        stopping_step = 0
        best_value = log_value
    else:
        stopping_step += 1

    if stopping_step >= flag_step:
        print("Early stopping at step: {} log:{}".format(flag_step, log_value))
        should_stop = True
    else:
        should_stop = False

    return best_value, stopping_step, should_stop

def train(model, data_generator, optimizer):
    """
    Train the model PyTorch style

    Arguments:
    ---------
    model: PyTorch model
    data_generator: Data object
    optimizer: PyTorch optimizer
    """
    model.train()
    n_batch = data_generator.n_train // data_generator.batch_size + 1
    running_loss=0
    for _ in range(n_batch):
        u, i, j = data_generator.sample()
        optimizer.zero_grad()
        loss = model(u,i,j)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss

def split_matrix(X, n_splits=100):
    """
    Split a matrix/Tensor into n_folds (for the user embeddings and the R matrices)

    Arguments:
    ---------
    X: matrix to be split
    n_folds: number of folds

    Returns:
    -------
    splits: split matrices
    """
    splits = []
    chunk_size = X.shape[0] // n_splits
    for i in range(n_splits):
        start = i * chunk_size
        end = X.shape[0] if i == n_splits - 1 else (i + 1) * chunk_size
        splits.append(X[start:end])
    return splits

def compute_ndcg_k(pred_items, test_items, test_indices, k, device):
    """
    Compute NDCG@k
    
    Arguments:
    ---------
    pred_items: binary tensor with 1s in those locations corresponding to the predicted item interactions
    test_items: binary tensor with 1s in locations corresponding to the real test interactions
    test_indices: tensor with the location of the top-k predicted items
    k: k'th-order 

    Returns:
    -------
    NDCG@k
    """
    r = (test_items * pred_items).gather(1, test_indices)
    f = torch.from_numpy(np.log2(np.arange(2, k+2))).float().to(device)
    dcg = (r[:, :k]/f).sum(1)
    dcg_max = (torch.sort(r, dim=1, descending=True)[0][:, :k]/f).sum(1)
    ndcg = dcg/dcg_max
    ndcg[torch.isnan(ndcg)] = 0
    return ndcg


def eval_model(u_emb, i_emb, Rtr, Rte, k, device):
    """
    Evaluate the model
    
    Arguments:
    ---------
    u_emb: User embeddings
    i_emb: Item embeddings
    Rtr: Sparse matrix with the training interactions
    Rte: Sparse matrix with the testing interactions
    k : kth-order for metrics
    
    Returns:
    --------
    result: Dictionary with lists correponding to the metrics at order k for k in Ks
    """
    # split matrices
    ue_splits = split_matrix(u_emb)
    tr_splits = split_matrix(Rtr)
    te_splits = split_matrix(Rte)

    recall_k, ndcg_k= [], []
    # compute results for split matrices
    for ue_f, tr_f, te_f in zip(ue_splits, tr_splits, te_splits):

        scores = torch.mm(ue_f, i_emb.t())

        test_items = torch.from_numpy(te_f.todense()).float().to(device)
        non_train_items = torch.from_numpy(1-(tr_f.todense())).float().to(device)
        scores = scores * non_train_items

        _, test_indices = torch.topk(scores, dim=1, k=k)
        pred_items = torch.zeros_like(scores).float()
        src = torch.ones(test_indices.size()).float().to(device)
        pred_items.scatter_(dim=1,index=test_indices,src=src)

        topk_preds = torch.zeros_like(scores).float()

        topk_preds.scatter_(dim=1,index=test_indices[:, :k],src=torch.ones_like(test_indices[:, :k], dtype=torch.float))

        TP = (test_items * topk_preds).sum(1)

        rec = torch.nan_to_num(TP/test_items.sum(1), nan=0)

        ndcg = compute_ndcg_k(pred_items, test_items, test_indices, k, device)

        recall_k.append(rec)
        ndcg_k.append(ndcg)

    return torch.cat(recall_k).mean(), torch.cat(ndcg_k).mean()

def probability_matrix(u_emb, i_emb, Rtr, Rte, device):
    """
    Compute link prediction matrix
    
    Arguments:
    ---------
    u_emb: User embeddings
    i_emb: Item embeddings
    Rtr: Sparse matrix with the training interactions
    Rte: Sparse matrix with the testing interactions
    device: Device to perform computation (CPU or GPU)
    
    Returns:
    --------
    prob_matrix: Matrix of interaction probabilities for each user-item pair
    """

    # split matrices
    ue_splits = split_matrix(u_emb)
    tr_splits = split_matrix(Rtr)
    te_splits = split_matrix(Rte)
    prob_matrix_list = []

    from scipy.sparse import find
    from collections import Counter
    non_zero_values = find(Rte)[2]
    value_counts = Counter(non_zero_values)
    print(value_counts)

    # compute results for split matrices
    for ue_f, tr_f, te_f in zip(ue_splits, tr_splits, te_splits):

        # Compute raw scores
        scores = torch.mm(ue_f, i_emb.t())

        # Convert scores to probabilities
        probabilities = torch.sigmoid(scores)

        # Filter out training interactions to focus on test interactions
        non_train_items = torch.from_numpy(1 - tr_f.todense()).float().to(device)
        scores = scores * non_train_items

        probabilities = probabilities * non_train_items

        # Store probabilities matrix for this split
        prob_matrix_list.append(probabilities.cpu().detach().numpy())

    # Combine probability matrices from all splits
    prob_matrix = np.concatenate(prob_matrix_list, axis=0)

    # Binarize prob_matrix
    predicted_ones = np.sum(prob_matrix > 0.9)
    correct_ones = np.sum((prob_matrix > 0.9) & (Rte.todense() > 0))
    actual_ones = np.sum(Rte.todense() > 0)

    print(predicted_ones, correct_ones, actual_ones, prob_matrix.shape, u_emb.shape, i_emb.shape, Rte.shape)
    precision = correct_ones / predicted_ones
    recall = correct_ones / actual_ones 
    f1_score = (2*precision*recall) / (precision + recall)

    return prob_matrix, recall, precision, f1_score