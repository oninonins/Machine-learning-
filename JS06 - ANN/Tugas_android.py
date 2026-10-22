import os
import sys
import time
import platform

print("=" * 60)
print("DEVICE_INFO")
print("platform:", platform.platform())
print("machine:", platform.machine())
print("processor:", platform.processor() or "-")
print("python:", sys.version.split()[0])
print("cpu_count:", os.cpu_count())


def _read(path):
    try:
        with open(path, encoding="utf-8", errors="ignore") as f:
            return f.read()
    except OSError:
        return ""


mem = _read("/proc/meminfo")
for line in mem.splitlines():
    if line.startswith("MemTotal:"):
        kb = int(line.split()[1])
        print("mem_total_mb:", round(kb / 1024))
        break

cpu = _read("/proc/cpuinfo")
seen = set()
for line in cpu.splitlines():
    if ":" not in line:
        continue
    key, val = line.split(":", 1)
    key = key.strip().lower()
    if key in ("hardware", "model name", "processor") and key not in seen:
        seen.add(key)
        print(f"{key}:", val.strip())

print("=" * 60)

import numpy as np
print("numpy:", np.__version__)

HAS_SKLEARN = False
try:
    import sklearn
    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import StandardScaler
    HAS_SKLEARN = True
    print("sklearn:", sklearn.__version__)
except Exception as exc:
    print("sklearn: TIDAK TERSEDIA ->", type(exc).__name__, exc)

HAS_ANNOY = False
try:
    from annoy import AnnoyIndex
    HAS_ANNOY = True
    print("annoy: tersedia")
except Exception as exc:
    print("annoy: TIDAK TERSEDIA ->", type(exc).__name__, exc)

HAS_HNSW = False
try:
    import hnswlib
    HAS_HNSW = True
    print("hnswlib: tersedia")
except Exception as exc:
    print("hnswlib: TIDAK TERSEDIA ->", type(exc).__name__, exc)

HAS_FAISS = False
try:
    import faiss
    HAS_FAISS = True
    print("faiss:", faiss.__version__)
except Exception as exc:
    print("faiss: TIDAK TERSEDIA ->", type(exc).__name__, exc)

print("=" * 60)

# -------------------------------
# Dataset (sama dengan notebook)
# -------------------------------
np.random.seed(42)
n_samples = 10000
d = 128
k = 10

# Kecilkan bila HP terlalu lambat, misal 2000.
# Nilai harus sama dengan Colab agar perbandingan adil.
N_QUERY = n_samples

X = np.random.random((n_samples, d)).astype(np.float32)

if HAS_SKLEARN:
    X_scaled = StandardScaler().fit_transform(X).astype(np.float32)
else:
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd == 0] = 1.0
    X_scaled = ((X - mu) / sd).astype(np.float32)

Q = X_scaled[:N_QUERY]
print(f"n_samples={n_samples} d={d} k={k} n_query={N_QUERY}")
print("=" * 60)

results = []


def exact_numpy(X, Q, k, chunk=256):
    x2 = (X * X).sum(axis=1)
    out = np.empty((len(Q), k), dtype=np.int32)
    for start in range(0, len(Q), chunk):
        q = Q[start:start + chunk]
        d2 = x2[start:start + chunk, None] - 2.0 * (q @ X.T) + x2[None, :]
        out[start:start + chunk] = np.argpartition(d2, k, axis=1)[:, :k]
        done = min(start + chunk, len(Q))
        print(f"  exact progress {done}/{len(Q)}", flush=True)
    return out


# -------------------------------
# Exact NN
# -------------------------------
print("[Exact NN]")
t0 = time.perf_counter()
if HAS_SKLEARN:
    nn = NearestNeighbors(n_neighbors=k, algorithm="brute", metric="euclidean")
    nn.fit(X_scaled)
    nn.kneighbors(Q)
else:
    exact_numpy(X_scaled, Q, k)
t_exact = time.perf_counter() - t0
results.append(("Exact NN", t_exact))
print(f"Exact NN done in {t_exact:.3f} s")

# -------------------------------
# KD-Tree dan Ball-Tree (pembanding lokal)
# -------------------------------
if HAS_SKLEARN:
    for algo in ("kd_tree", "ball_tree"):
        print(f"[{algo}]")
        try:
            t0 = time.perf_counter()
            nn = NearestNeighbors(n_neighbors=k, algorithm=algo, metric="euclidean")
            nn.fit(X_scaled)
            nn.kneighbors(Q)
            elapsed = time.perf_counter() - t0
            results.append((algo, elapsed))
            print(f"{algo} done in {elapsed:.3f} s")
        except Exception as exc:
            print(f"{algo} GAGAL ->", type(exc).__name__, exc)

# -------------------------------
# Annoy
# -------------------------------
if HAS_ANNOY:
    print("[Annoy]")
    t0 = time.perf_counter()
    index_annoy = AnnoyIndex(d, "euclidean")
    for i, v in enumerate(X_scaled):
        index_annoy.add_item(i, v)
    index_annoy.build(10)
    for v in Q:
        index_annoy.get_nns_by_vector(v, k)
    t_annoy = time.perf_counter() - t0
    results.append(("Annoy", t_annoy))
    print(f"Annoy done in {t_annoy:.3f} s")
else:
    print("[Annoy] dilewati")

# -------------------------------
# HNSW
# -------------------------------
if HAS_HNSW:
    print("[HNSW]")
    t0 = time.perf_counter()
    p_hnsw = hnswlib.Index(space="l2", dim=d)
    p_hnsw.init_index(max_elements=n_samples, ef_construction=200, M=16)
    p_hnsw.add_items(X_scaled)
    p_hnsw.set_ef(200)
    p_hnsw.knn_query(Q, k=k)
    t_hnsw = time.perf_counter() - t0
    results.append(("HNSW", t_hnsw))
    print(f"HNSW done in {t_hnsw:.3f} s")
else:
    print("[HNSW] dilewati")

# -------------------------------
# FAISS IVF
# -------------------------------
if HAS_FAISS:
    print("[FAISS IVF]")
    t0 = time.perf_counter()
    quantizer = faiss.IndexFlatL2(d)
    index_faiss = faiss.IndexIVFFlat(quantizer, d, 100, faiss.METRIC_L2)
    index_faiss.train(X_scaled)
    index_faiss.add(X_scaled)
    index_faiss.nprobe = 10
    index_faiss.search(Q, k)
    t_faiss = time.perf_counter() - t0
    results.append(("FAISS IVF", t_faiss))
    print(f"FAISS IVF done in {t_faiss:.3f} s")
else:
    print("[FAISS IVF] dilewati")

# -------------------------------
# Ringkasan
# -------------------------------
print("=" * 60)
print("RESULT_TABLE")
print(f"{'Metode':<12}{'Waktu (s)':>12}")
for name, elapsed in results:
    print(f"{name:<12}{elapsed:>12.3f}")

print("=" * 60)
print("MACHINE_READABLE")
for name, elapsed in results:
    print(f"RESULT|{name}|{elapsed:.4f}")
print("DONE")
