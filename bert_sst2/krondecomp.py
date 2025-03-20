import pickle
import numpy as np
from scipy.sparse.linalg import svds, LinearOperator


with open("sst2_grads.pickle", "rb") as fp:   #Pickling
    dict_of_grads = pickle.load(fp)

def is_pos_def(x):
    return np.all(np.linalg.eigvals(x) > 0)


def get_kron_factors(list_of_grads, layer):
    m, n = list_of_grads[0].shape #cols rows in torch -> rows cols in numpy
    print ("m, n", m, n)
    list_of_grads1 = [grad.reshape(-1).numpy() for grad in list_of_grads]

    grad_vectors = np.stack([grad.reshape(n,m, order = 'F') for grad in list_of_grads1])

    def matvec(vec):
        print ('matvec', flush = True)
        k, m, n = grad_vectors.shape
        V = vec.reshape(m, m, order='F')
        res = np.zeros(n*n) 
        for i in range(k):
            res += (grad_vectors[i].T @ V @ grad_vectors[i]).T.ravel()
        return res/k
    
    def r_matvec(vec):
        print ('rmatvec', flush = True)
        k, m, n = grad_vectors.shape
        V = vec.reshape(n, n, order='F')
        res = np.zeros(m*m) 
        for i in range(k):
            res += (grad_vectors[i] @ V @ grad_vectors[i].T).T.ravel()
        return res/k

    linop_m = LinearOperator(
    shape=(m**2, n**2),
    matvec=matvec,
    rmatvec=r_matvec)

    # Compute SVD
    print("Performing SVD...", flush=True)
    u, s, vt = svds(linop_m, k = 3, return_singular_vectors=True)
    sidx = np.argsort(-s)
    s = s[sidx]
    u = u[:, sidx]
    v = vt[sidx, :].T

    print ("s", s)
    x_ = u[:, 0] * s[0]
    y_ = v[:, 0]
    XF = x_.reshape(m, m, order='F')
    YF = y_.reshape(n, n, order='F')

    print (is_pos_def(YF))

    np.save("/home/jovyan/shares/SR004.nfs2/chekalina/FisherKronecker/bert_sst2/kron_factors/C1sgd_"+str(key)+".npy", XF)
    np.save("/home/jovyan/shares/SR004.nfs2/chekalina/FisherKronecker/bert_sst2/kron_factors/B1sgd_"+str(key)+".npy", YF)

    return 0

part_of = ["bert.encoder.layer.9.output.dense", "bert.encoder.layer.10.output.dense", "bert.encoder.layer.11.output.dense"]

for key in part_of:
    print ("layer ", key, flush = True)
    list_of_grads = dict_of_grads[key]
    get_kron_factors(list_of_grads, key)
    print ("finished collected gradients in layer ", key, flush = True)