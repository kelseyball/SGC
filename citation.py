import time
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from utils import load_citation, sgc_precompute, set_seed
from models import get_model
from metrics import accuracy
import pickle as pkl
from args import get_citation_args
from time import perf_counter

# Arguments
args = get_citation_args()

if args.tuned:
    if args.model == "SGC":
        with open("{}-tuning/{}.txt".format(args.model, args.dataset), 'rb') as f:
            args.weight_decay = pkl.load(f)['weight_decay']
            print("using tuned weight decay: {}".format(args.weight_decay))
    else:
        raise NotImplemented

# setting random seeds
set_seed(args.seed, args.cuda)

eadj, features, labels, idx_train, idx_val, idx_test = load_citation(args.dataset, args.cuda)
print(f'eadj.shape {eadj.shape}')
print(f'features.shape {features.shape}')
print(f'labels.shape {labels.shape}')

model = get_model(
    model_opt=args.model,
    nfeat=features.size(1),
    nclass=labels.max().item()+1,
    nhid=args.hidden,
    dropout=args.dropout,
    cuda=args.cuda
)

if args.model == "SGC":
    features, precompute_time = sgc_precompute(
        features=features,
        adj=eadj,
        degree=args.degree
    )
    print("{:.4f}s".format(precompute_time))

def train_regression(args,
                     model,
                     features,
                     labels,
                     idx_train,
                     idx_val):

    optimizer = optim.Adam(model.parameters(), lr=args.lr,
                           weight_decay=args.weight_decay)
    t = perf_counter()
    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        output = model(features)
        loss_train = F.cross_entropy(output[idx_train], labels[idx_train])
        loss_train.backward()
        optimizer.step()
    train_time = perf_counter()-t

    with torch.no_grad():
        model.eval()
        output = model(features)
        acc_val = accuracy(output[idx_val], labels[idx_val])

    return model, acc_val, train_time

def test_regression(model, features, labels, idx_test):
    model.eval()
    output = model(features)
    return accuracy(output[idx_test], labels[idx_test])

if args.model == "SGC":
    model, acc_val, train_time = train_regression(
        args=args,
        model=model,
        features=features,
        labels=labels,
        idx_train=idx_train,
        idx_val=idx_val,
    )
    acc_test = test_regression(
        model=model,
        features=features,
        labels=labels,
        idx_test=idx_test
    )

print("Validation Accuracy: {:.4f} Test Accuracy: {:.4f}".format(acc_val, acc_test))
print("Pre-compute time: {:.4f}s, train time: {:.4f}s, total: {:.4f}s".format(precompute_time, train_time, precompute_time+train_time))
