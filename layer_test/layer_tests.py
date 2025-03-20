import json
import torch, torch.nn as nn
import torch.nn.functional as F

import numpy as np
from cifar import load_cifar10
import argparse

import pickle

def is_pos_def(x):
    return np.all(np.linalg.eigvals(x) > 0)

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


def iterate_minibatches(X, y, batchsize):
    indices = np.random.permutation(np.arange(len(X)))
    for start in range(0, len(indices), batchsize):
        ix = indices[start: start + batchsize]
        yield X[ix], y[ix]

def evalm(model, X_test, y_test):
    model.train(False) # disable dropout / use averages for batch_norm
    test_batch_acc = []
    for X_batch, y_batch in iterate_minibatches(X_test, y_test, 500):
        logits = model(torch.as_tensor(X_batch, dtype=torch.float32))
        y_pred = logits.max(1)[1].data.numpy()
        test_batch_acc.append(np.mean(y_batch == y_pred))

    test_accuracy = np.mean(test_batch_acc)

    return test_accuracy


def factorize_to_fwsvd(module, fc_b, avg_grads, rank):
    I = torch.diag(torch.sqrt(avg_grads.sum(0))).to(module.weight.device, module.weight.dtype)

    U, S, Vt = torch.linalg.svd((I @ module.weight.T).T.to(module.weight.device) , full_matrices=False) # driver='gesvdj'

    w1 = torch.linalg.lstsq(I, torch.mm(torch.diag(torch.sqrt(S[0:rank])),Vt[0:rank, :]).T).solution.T
    w2 = torch.mm(U[:, 0:rank], torch.diag(torch.sqrt(S[0:rank])))

    # create new layers and insert weights
    fc_w = module.weight.data.cpu().data.numpy()
    out_features, in_features = fc_w.shape
    is_bias = fc_b is not None

    linear1 = nn.Linear(in_features = in_features,
                      out_features = rank,
                      bias = False)
    linear1.weight = nn.Parameter(torch.FloatTensor(w1))

    linear2 = nn.Linear(in_features = rank,
                      out_features = out_features,
                      bias=is_bias)
    linear2.weight = nn.Parameter(torch.FloatTensor(w2))
    linear2.bias = nn.Parameter(torch.FloatTensor(fc_b))
    print (w1.shape, w2.shape)
    # create factorized layer
    factorized_layer = nn.Sequential(linear1,linear2)
    return factorized_layer

def factorize_to_svd(fc_w, fc_b, rank):
    U, S, Vt = np.linalg.svd(fc_w, full_matrices=False)
    # truncate SVD and fuse Sigma matrix
    w1 = np.dot(np.diag(np.sqrt(S[0:rank])),Vt[0:rank, :])
    w2 = np.dot(U[:,0:rank,], np.diag(np.sqrt(S[0:rank])))

    # create new layers and insert weights
    out_features, in_features = fc_w.shape
    is_bias = fc_b is not None

    linear1 = nn.Linear(in_features = in_features,
                      out_features = rank,
                      bias = False)
    linear1.weight = nn.Parameter(torch.FloatTensor(w1))

    linear2 = nn.Linear(in_features = rank,
                      out_features = out_features,
                      bias=is_bias)
    linear2.weight = nn.Parameter(torch.FloatTensor(w2))
    linear2.bias = nn.Parameter(torch.FloatTensor(fc_b))
    print (w1.shape, w2.shape)
    # create factorized layer
    factorized_layer = nn.Sequential(linear1,linear2)
    return factorized_layer

def factorize_to_kron_svd(fc_w, fc_b, B11, C11, rank):
    alpha = 0.0
    B_new = B11
    while (not is_pos_def(B_new)):
        alpha += 0.1
        B_new = (1 - alpha)*B11  + alpha*np.eye(len(np.diag(B11)))

    print ("alpha",alpha)
    alpha = 0.0
    C_new = C11
    while (not is_pos_def(C_new)):
        alpha += 0.1
        C_new = (1 - alpha)*C11  + alpha*np.eye(len(np.diag(C11)))
    print ("alpha",alpha)

    B1_square = np.linalg.cholesky(B_new)
    C1_square = np.linalg.cholesky(C_new)
    U, S, Vt = np.linalg.svd(C1_square.T@fc_w@B1_square, full_matrices=False)

    U1 = np.linalg.inv(C1_square.T)@U
    V1t = Vt@np.linalg.inv(B1_square)

    w1 = np.diag(np.sqrt(S[:rank]))@V1t[:rank, :]
    w2 = U1[:,:rank] @ np.diag(np.sqrt(S[:rank]))

    out_features, in_features = fc_w.shape
    is_bias = fc_b is not None

    linear1 = nn.Linear(in_features = in_features,
                      out_features = rank,
                      bias = False)
    linear1.weight = nn.Parameter(torch.FloatTensor(w1))

    linear2 = nn.Linear(in_features = rank,
                      out_features = out_features,
                      bias=is_bias)
    linear2.weight = nn.Parameter(torch.FloatTensor(w2))
    linear2.bias = nn.Parameter(torch.FloatTensor(fc_b))

    factorized_layer = nn.Sequential(linear1,linear2)

    return factorized_layer

def create_and_load_model():
    model = nn.Sequential()
    model.add_module('flatten', Flatten())
    model.add_module('dense1', nn.Linear(3072,2304))
    model.add_module('dense1_relu', nn.ReLU())
    
    model.add_module('dense2', nn.Linear(2304, 768))
    model.add_module('dense2_relu', nn.ReLU())
    
    model.add_module('dense3', nn.Linear(768, 32*6))
    model.add_module('dense3_relu', nn.ReLU())
    model.add_module('dense4', nn.Linear(32*6, 64))
    model.add_module('dense4_relu', nn.ReLU())
    model.add_module('dense5_logits', nn.Linear(64, 10)) 

    model.load_state_dict(torch.load("./state_dict.pth"))

    return model
    


def compress_model(model, list_of_layer_to_compress, rank_ratio, method):

    for layer in list_of_layer_to_compress:
        if layer == '3':
            fc_w = model.dense3
            fc_b = model.dense3
        elif layer == '2':
            fc_w = model.dense2
            fc_b = model.dense2
        elif layer == '4':
            fc_w = model.dense4
            fc_b = model.dense4

        rank = int(round(fc_w.weight.data.cpu().data.numpy().shape[0]*rank_ratio))
        if method == 'svd':
            factorized_layer = factorize_to_svd(fc_w.weight.data.cpu().data.numpy(), fc_w.bias.data.cpu().data.numpy(), rank)
        elif method == 'fwsvd':
            with open("list_of_grads"+str(layer), "rb") as fp:   #Pickling
                list_of_grads = pickle.load(fp)
            list_of_grads_pow = [torch.pow(elem,2) for elem in list_of_grads]
            avg_grads = torch.mean(torch.stack(list_of_grads_pow, dim=0),dim = 0)
            factorized_layer = factorize_to_fwsvd(fc_w, fc_w.bias.data.cpu().data.numpy(), avg_grads, rank)
            
        elif method == 'kron':
            B11 = np.load("./small_factors/B1_"+str(layer) +".npy")#np.linalg.cholesky(B1)
            C11 = np.load("./small_factors/C1_"+str(layer) +".npy")#np.linalg.cholesky(C1)
            factorized_layer = factorize_to_kron_svd(fc_w.weight.data.cpu().data.numpy(), fc_w.bias.data.cpu().data.numpy(), B11, C11, rank)

        if layer == '3':
            model.dense3 = factorized_layer
        elif layer == '2':
            model.dense2 = factorized_layer
        elif layer == '4':
            model.dense4 = factorized_layer
    

    return model





def main():
    parser = argparse.ArgumentParser(
        description="Generate experiment folder structure."
    )
    parser.add_argument(
        "--direction", type=str, required=True, help="Tuda or Obratno."
    )

    args = parser.parse_args()

    compression_sequences = {}
    compression_sequences['tuda'] = [['2'], ['2', '3'],  ['2', '3', '4']]
    compression_sequences['one'] = [['2'], ['3'],  ['4']]
    compression_sequences['obratno'] = [['4'], ['3', '4'], ['4','2', '3']]

    ranks_ratio = [0.01, 0.05, 0.1, 0.5]

    seqs = compression_sequences[args.direction]

    results = {"svd":{}, "fwsvd":{}, "kron":{}}

    X_train, y_train, X_val, y_val, X_test, y_test = load_cifar10("cifar_data")

    for method in ["svd","fwsvd","kron"]:
        for seq in seqs:
            seq_str = ''.join(seq)
            results[method][seq_str] = []
            for rank_r in ranks_ratio:
                print (method, seq, rank_r)
                model = create_and_load_model()
                print ("model created")
                model = compress_model(model, seq, rank_r, method) 
                print ("model compressed")
                score = evalm(model, X_test, y_test)
                print ("score", score)
                results[method][seq_str].append(score)

        print ("\nresults!!!\n", results)
                

    with open('results'+str(args.direction)+'.json', 'w') as fp:
        json.dump(results, fp)



if __name__ == "__main__":
    main()