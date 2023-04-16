import os
import numpy as np

import torch
import torch.utils
import torch.utils.data

import torchvision
from tqdm import tqdm
import argparse

class MahalanobisClassifier:
    """
    X: training set data points [n_samples, n_dim]
    y: labels [n_samples,]
    n_cls: number of classes
    """
    def __init__(self, X, y, n_cls, verbose=True):
        self.n_cls = n_cls
        self.n_dim = X.shape[1]
        self.mu = np.zeros((n_cls, self.n_dim)).astype(np.float32)
        self.sigma = np.zeros((n_cls, self.n_dim, self.n_dim)).astype(np.float32)
        self.count = np.zeros(n_cls).astype(np.float32)

        pbar = tqdm(range(X.shape[0]), desc='Initializing Mahalanobis Classifier') if verbose else range(X.shape[0])
        for i in pbar:
            self.mu[y[i]] += X[i]
            x_kpdim = X[i:i+1]
            self.sigma[y[i]] += x_kpdim.T @ x_kpdim
            self.count[y[i]] += 1
        
        for i in range(self.n_cls):
            self.mu[i] /= self.count[i]
            mui_kpdim = self.mu[i:i+1] # (1, n_dim)
            self.sigma[i] = self.sigma[i] / self.count[i] - mui_kpdim.T @ mui_kpdim
        
        sigma_avg = self.sigma.sum(axis=0) / self.n_cls
        self.sigma_inv = np.linalg.inv(sigma_avg)
    
    def compute_dist(self, x):
        # x: [1, n_dim]
        assert x.ndim == 2 and x.shape[0] == 1, f'invalid one-sample shape {x.shape}'
        dists = []
        for i in range(self.n_cls):
            x_delta = x - self.mu[i:i+1] # (1, n_dim)
            dist = (x_delta @ self.sigma_inv) @ x_delta.T
            dists.append(dist)
        dists = np.array(dists)
        return dists
    
    def predict(self, X, return_dists=False):
        # X: [n_samples, n_dim]
        n_samples = X.shape[0]
        all_dists, predicts = [], []
        for i in range(n_samples):
            dists = self.compute_dist(X[i:i+1])
            all_dists.append(dists)
            predicts.append(dists.argmin())

        predicts = np.array(predicts)

        if not return_dists:
            return predicts
        else:
            all_dists = np.array(all_dists)
            return predicts, all_dists

def parse_args():
    parser = argparse.ArgumentParser(description="Minimum Mahalanobis Distance Classifier with Dimension Reduction")
    parser.add_argument(
        "--data",
        type=str,
        default='mnist',
        choices=['mnist', 'cifar10', 'cifar100'],
        help="Dataset used to evaluate the algorithm.",
    )
    parser.add_argument(
        '--reduction',
        type=str,
        default='pca',
        choices=['pca', 'lda', 'no'],
        help='Dimension reduction method applied.'
    )
    parser.add_argument(
        '--n_dim',
        type=int,
        default=128,
        help='Dimension after reduction.'
    )
    parser.add_argument(
        '--grid_exp',
        action='store_true',
        help='Whether to use grid experiments to compute all exp values.'
    )
    args = parser.parse_args()
    return args

def load_all_data(dataset):
    dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=len(dataset), num_workers=8)
    images, labels = next(iter(dataloader))
    images = images.flatten(start_dim=1)
    return images.numpy(), labels.numpy()

def get_accuracy(pred, gt):
    # pred, gt: [n_samples, ]
    assert pred.shape[0] == gt.shape[0], f'Samples count not the same, pred:{pred.shape[0]} , gt:{gt.shape[0]}'
    return (pred == gt).sum() / gt.shape[0]

def main(args, verbose=True):
    data_root = os.path.join('data', args.data)
    n_channel = 1 if args.data == 'mnist' else 3
    data_transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.5] * n_channel, std=[0.5] * n_channel)
    ])
    if args.data == 'mnist':
        train_data = torchvision.datasets.MNIST(root=data_root, train=True, download=True, transform=data_transform)
        test_data = torchvision.datasets.MNIST(root=data_root, train=False, download=True, transform=data_transform)
    elif args.data == 'cifar10':
        train_data = torchvision.datasets.CIFAR10(root=data_root, train=True, download=True, transform=data_transform)
        test_data = torchvision.datasets.CIFAR10(root=data_root, train=False, download=True, transform=data_transform)
    elif args.data == 'cifar100':
        train_data = torchvision.datasets.CIFAR100(root=data_root, train=True, download=True, transform=data_transform)
        test_data = torchvision.datasets.CIFAR100(root=data_root, train=False, download=True, transform=data_transform)
    n_classes = len(train_data.classes)

    # load data
    trainX, trainY = load_all_data(train_data)
    testX, testY = load_all_data(test_data)

    # dimension reduction transformer
    if args.reduction == 'no':
        trainX_red, testX_red = trainX, testX
    else:
        print(f'Initializing dimension reduction transformer {args.reduction}')
        if args.reduction == 'pca':
            from sklearn.decomposition import PCA
            transformer = PCA(n_components=args.n_dim)
            transformer.fit(trainX)
        else:
            from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
            transformer = LinearDiscriminantAnalysis(n_components=args.n_dim)
            transformer.fit(trainX, trainY)
        
        exp_var = sum(transformer.explained_variance_ratio_)
        print(f'Explained Var Ratio {exp_var:.3f}')

        print(f'Reducing train data ...')
        trainX_red = transformer.transform(trainX)
        print(f'Reducing test data ...')
        testX_red = transformer.transform(testX)

    # Minimum Mahalanobis Distance Classifier training
    print(f'Initializing Minimum Mahalanobis Distance Classifier ...')
    classifier = MahalanobisClassifier(trainX_red, trainY, n_cls=n_classes, verbose=verbose)

    # Evaluation
    print(f'Evaluating train set ...')
    pred_train = classifier.predict(trainX_red)
    acc_train = get_accuracy(pred_train, trainY)
    if verbose:
        print(f'Train set accuracy {acc_train*100:.2f}%')

    print(f'Evaluating test set ...')
    pred_test = classifier.predict(testX_red)
    acc_test = get_accuracy(pred_test, testY)
    if verbose:
        print(f'Test set accuracy {acc_test*100:.2f}%')

    return acc_train * 100, acc_test * 100, exp_var

def grid_run():
    from argparse import Namespace

    grid_dict = {
        'mnist': {'ncls':10, 'dim': [2, 4, 6, 8, 9, 20, 50, 100, 300, 500]},
        'cifar10': {'ncls':10, 'dim': [2, 4, 6, 8, 9, 20, 50, 100, 300, 500]},
        'cifar100': {'ncls':100, 'dim': [5, 10, 20, 40, 60, 80, 99, 200, 300, 500]}
    }

    # grid_dict = {
    #     'mnist': {'ncls':10, 'dim': [2, 20]},
    #     'cifar100': {'ncls':100, 'dim': [60, 200]}
    # }

    total_exps = 0
    for data in grid_dict.keys():
        ncls = grid_dict[data]['ncls']
        for ndim in grid_dict[data]['dim']:
            total_exps += 1 # pca
            if ndim < ncls:
                total_exps += 1 # lda
    
    pbar = tqdm(total=total_exps, desc='Grid Experiment')

    outd = {
        'data': [],
        'ndim': [],
        'pca_train': [],
        'pca_test': [],
        'pca_exp_var': [],
        'lda_train': [],
        'lda_test': [],
        'lda_exp_var': []
    }

    for data in grid_dict.keys():
        ncls = grid_dict[data]['ncls']
        for ndim in grid_dict[data]['dim']:
            args = {'data':data, 'reduction':'pca', 'n_dim':ndim}
            args = Namespace(**args)
            pca_train, pca_test, pca_var = main(args, verbose=False)
            pbar.update(1)

            if ndim < ncls:
                args = {'data':data, 'reduction':'lda', 'n_dim':ndim}
                args = Namespace(**args)
                lda_train, lda_test, lda_var = main(args, verbose=False)
                pbar.update(1)
            else:
                lda_train, lda_test, lda_var = 0, 0, 0
            
            entry = [data, ndim, f'{pca_train:.2f}', f'{pca_test:.2f}', f'{pca_var:.3f}', f'{lda_train:.2f}', f'{lda_test:.2f}', f'{lda_var:.3f}']
            for i, key in enumerate(outd.keys()):
                outd[key].append(entry[i])

    import pandas as pd
    df = pd.DataFrame(data=outd)
    print(df)
    df.to_csv('exp.csv', index=False)

if __name__ == '__main__':
    args = parse_args()
    if args.grid_exp:
        grid_run()
    else:
        main(args)        



