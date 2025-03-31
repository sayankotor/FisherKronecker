import pickle
import cupy as cp
from cupyx.scipy.sparse.linalg import svds, LinearOperator
import time
import concurrent.futures
import traceback

def get_kron_factors(list_of_grads, top_k=10, layer_name="linear", device_id=0):
    def r_matvec(vec, grad_vectors):
        k, m, n = grad_vectors.shape
        V = vec.reshape(m, m, order="F")
        GTVG = grad_vectors.transpose(0, 2, 1) @ V @ grad_vectors
        return cp.mean(GTVG, axis=0).T.ravel()

    def matvec(vec, grad_vectors):
        k, m, n = grad_vectors.shape
        V = vec.reshape(n, n, order="F")
        GVGT = grad_vectors @ V @ grad_vectors.transpose(0, 2, 1)
        return cp.mean(GVGT, axis=0).T.ravel()

    # Get available GPUs
    num_devices = cp.cuda.runtime.getDeviceCount()
    device_pool = [cp.cuda.Device(i) for i in range(num_devices)]
    with device_pool[device_id]:
        m, n = list_of_grads[0].shape
        grad_vectors = cp.stack([cp.asarray(grad).reshape(m, n, order="F") for grad in list_of_grads])  # Shape: (k, m, n)
        print(layer_name, grad_vectors.shape)
        linop = LinearOperator(
            shape=(m * m, n * n),
            matvec=lambda vec: matvec(vec, grad_vectors),
            rmatvec=lambda vec: r_matvec(vec, grad_vectors),
            dtype=cp.float32,
        )

        u, s, vt = svds(linop, k=top_k, return_singular_vectors=True)
        sidx = cp.argsort(-s)
        s = s[sidx]
        u = u[:, sidx]
        v = vt[sidx, :].T

        XF = (u[:, 0] * s[0]).reshape(m, m, order="F")
        YF = v[:, 0].reshape(n, n, order="F")

        print(f"✔ Layer {layer_name} on device {device_id} done | top singular value: {s[0]:.4f}")
        return XF, YF


# Target layers
target_layers = [
    "bert.encoder.layer.1.intermediate.dense",
    "bert.encoder.layer.2.intermediate.dense",
    "bert.encoder.layer.3.intermediate.dense",
    "bert.encoder.layer.4.intermediate.dense",
    "bert.encoder.layer.5.intermediate.dense",
    "bert.encoder.layer.6.intermediate.dense",
    "bert.encoder.layer.7.intermediate.dense",
    "bert.encoder.layer.8.intermediate.dense",
    "bert.encoder.layer.9.intermediate.dense",
    "bert.encoder.layer.10.intermediate.dense",
    "bert.encoder.layer.11.intermediate.dense",
    "bert.encoder.layer.1.output.dense",
    "bert.encoder.layer.2.output.dense",
    "bert.encoder.layer.3.output.dense",
    "bert.encoder.layer.4.output.dense",
    "bert.encoder.layer.5.output.dense",
    "bert.encoder.layer.6.output.dense",
    "bert.encoder.layer.7.output.dense",
    "bert.encoder.layer.8.output.dense",
    "bert.encoder.layer.9.output.dense",
    "bert.encoder.layer.10.output.dense",
    "bert.encoder.layer.11.output.dense",
]

# Load gradients (ensure they're already saved in GPU-compatible format)
with open("/workspace-SR004.nfs2/chekalina/FisherKronecker/bert_cola/cola_grads_v2.pickle", "rb") as fp:
    dict_of_grads = pickle.load(fp)

# target_layers = list(dict_of_grads.keys())

# num_devices = cp.cuda.runtime.getDeviceCount()
num_devices = 1
# Parallel execution using ThreadPoolExecutor (safe for GPU context)
start = time.time()
results = {}
top_k = 10
with concurrent.futures.ThreadPoolExecutor(max_workers=len(target_layers)) as executor:
    futures = []
    for i, key in enumerate(target_layers):
        grads = dict_of_grads[key]
        device_id = i % num_devices
        futures.append(executor.submit(get_kron_factors, grads, top_k, key, device_id))

    for future, key in zip(futures, target_layers):
        try:
            results[key] = future.result()
        except Exception as e:
            print(f"❌ Layer {key} failed with error: {e}")
            tb_str = traceback.format_exc()
            print("Full traceback as string:\n", tb_str)

end = time.time()
print(f"\n🕒 Total time: {end - start:.2f} seconds")
