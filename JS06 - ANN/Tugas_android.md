# Tugas JS06 ANN — Perbandingan Colab vs Android (Pydroid 3)

Perbandingan waktu komputasi Exact NN, Annoy, HNSW, dan FAISS IVF pada dataset acak 10.000 vektor berdimensi 128, dijalankan di dua lingkungan berbeda.

Kode Android tersedia di [`Tugas_android.py`](Tugas_android.py). Kode Colab tersedia di [`Tugas.ipynb`](Tugas.ipynb).

## 1. Spesifikasi Perangkat

| Item | Android (Pydroid 3) | Google Colab |
|---|---|---|
| CPU | MediaTek Dimensity 7046-Ultra, octa-core, max 2.5 GHz | Tidak diketahui (runtime cloud) |
| RAM | 7.5 GB (7.486 MB terdeteksi) | Tidak diketahui |
| Arsitektur | `aarch64` | `x86_64` |
| OS | Android 16 (kernel `6.12.30-android16-5`) | Linux |
| Python | 3.13.13 | 3.13 |
| NumPy | 2.4.4 | 2.1.3 |
| scikit-learn | 1.8.0 | Tersedia |
| Annoy | Tersedia | Tersedia |
| hnswlib | Tersedia | Tersedia |
| FAISS | **Tidak tersedia** (`ModuleNotFoundError`) | Tersedia (`faiss-cpu 1.15.1`) |

## 2. Kode Android

Kode lengkap ada di `Tugas_android.py`. Alur utamanya:

1. Cetak informasi perangkat dari `platform`, `/proc/cpuinfo`, dan `/proc/meminfo`.
2. Import `numpy`, `scikit-learn`, `annoy`, `hnswlib`, dan `faiss` di dalam `try/except`, lalu catat mana yang tersedia.
3. Buat dataset `np.random.seed(42)`, 10.000 × 128, `float32`.
4. Standarisasi dengan `StandardScaler`.
5. Ukur Exact NN, `kd_tree`, `ball_tree`, Annoy, HNSW, dan FAISS IVF.
6. Cetak tabel ringkasan dan output `RESULT|nama|detik`.

Cakupan waktu sama dengan notebook Colab: Exact = `fit` + query; Annoy = `add_item` + `build` + query; HNSW = `init_index` + `add_items` + query; FAISS = `train` + `add` + query.

## 3. Output Android (Pydroid 3)

```
============================================================
DEVICE_INFO
platform: Linux-6.12.30-android16-5-g1ed949324a3e-ab13881345-4k-aarch64-with-libc
machine: aarch64
processor: -
python: 3.13.13
cpu_count: 8
mem_total_mb: 7486
processor: 0
============================================================
numpy: 2.4.4
sklearn: 1.8.0
annoy: tersedia
hnswlib: tersedia
faiss: TIDAK TERSEDIA -> ModuleNotFoundError No module named 'faiss'
============================================================
n_samples=10000 d=128 k=10 n_query=10000
============================================================
[Exact NN]
Exact NN done in 1.092 s
[kd_tree]
kd_tree done in 51.479 s
[ball_tree]
ball_tree done in 49.340 s
[Annoy]
Annoy done in 1.475 s
[HNSW]
HNSW done in 8.267 s
[FAISS IVF] dilewati
============================================================
RESULT_TABLE
Metode         Waktu (s)
Exact NN           1.092
kd_tree           51.479
ball_tree         49.340
Annoy              1.475
HNSW               8.267
============================================================
MACHINE_READABLE
RESULT|Exact NN|1.0919
RESULT|kd_tree|51.4789
RESULT|ball_tree|49.3402
RESULT|Annoy|1.4751
RESULT|HNSW|8.2669
DONE
```

## 4. Output Colab

```
=== Ringkasan Waktu (detik) ===
Exact NN : 1.339
Annoy    : 0.634
HNSW     : 5.346
FAISS    : 0.609

     Metode  Waktu (s)
0   Exact NN   1.338962
1      Annoy   0.634222
2       HNSW   5.345786
3  FAISS IVF   0.609038
```

## 5. Tabel Perbandingan

| Metode | Colab (s) | Android (s) | Android / Colab |
|---|---:|---:|---:|
| Exact NN | 1.3390 | 1.0919 | 0.82x (Android lebih cepat) |
| Annoy | 0.6342 | 1.4751 | 2.33x lebih lambat |
| HNSW | 5.3458 | 8.2669 | 1.55x lebih lambat |
| FAISS IVF | 0.6090 | tidak tersedia | — |
| kd_tree | tidak dijalankan | 51.4789 | — |
| ball_tree | tidak dijalankan | 49.3402 | — |

Urutan tercepat:

- **Colab:** FAISS IVF (0.609) → Annoy (0.634) → Exact NN (1.339) → HNSW (5.346)
- **Android:** Exact NN (1.092) → Annoy (1.475) → HNSW (8.267); `kd_tree` dan `ball_tree` jauh di belakang.

## 6. Analisis

1. **Pada skala 10.000 × 128, Exact NN sangat kompetitif.** Di Android, brute force menjadi metode tercepat (1.092 s), lebih cepat dari Annoy (1.475 s) dan HNSW (8.267 s). Biaya build index ANN belum terbayar pada jumlah data dan jumlah query ini.

2. **Exact NN di Android 18% lebih cepat daripada Colab** (1.0919 s vs 1.338962 s). Android memakai 8 core `aarch64`. Spesifikasi CPU Colab tidak tercatat pada output, sehingga penyebab pasti selisih ini tidak dapat ditentukan dari data yang ada. Selisih juga hanya berasal dari satu kali run, jadi belum cukup untuk menyimpulkan perbedaan performa CPU secara umum.

3. **Annoy di Android 2.33x lebih lambat daripada Colab** (1.4751 s vs 0.6342 s). Annoy membangun dan mencari tree secara single-thread, jadi penambahan jumlah core tidak membantu. Perbedaan ini lebih mencerminkan performa per-core dan frekuensi CPU, bukan jumlah core.

4. **HNSW di Android 1.55x lebih lambat** (8.2669 s vs 5.3458 s). Waktu HNSW didominasi proses build graph (`ef_construction=200`, `M=16`) yang jauh lebih berat daripada build Annoy. Query-nya cepat, tetapi biaya build baru terbayar jika jumlah query sangat banyak atau data jauh lebih besar.

5. **FAISS IVF paling cepat di Colab (0.6090 s) tetapi tidak tersedia di Android.** `faiss-cpu` tidak memiliki wheel untuk Pydroid, sehingga `import faiss` gagal. Ini batasan praktis, bukan kesalahan kode.

6. **`kd_tree` dan `ball_tree` tidak cocok untuk 128 dimensi.** Keduanya 45–47x lebih lambat daripada brute force (51.4789 s dan 49.3402 s vs 1.0919 s). Pada dimensi tinggi, pruning tree tidak efektif karena hampir semua cabang tetap perlu diperiksa (*curse of dimensionality*).

7. **Perbandingan antarperangkat tidak sepenuhnya apple-to-apple.** Hardware Colab tidak diketahui, jumlah core berbeda, dan Android berpotensi mengalami thermal throttling. Waktu juga mencakup build + seluruh query, bukan query saja.

## 7. Kesimpulan

- Untuk dataset 10.000 × 128 dengan 10.000 query, **Exact NN adalah pilihan terbaik di Android** dan masih kompetitif di Colab.
- **Annoy** menjadi alternatif ANN paling praktis di Android karena tersedia dan hanya 1.35x lebih lambat daripada Exact NN.
- **HNSW** tersedia di Android, tetapi biaya build-nya membuatnya paling lambat pada skenario ini.
- **FAISS** unggul di Colab, tetapi belum bisa dipakai di Pydroid 3.
- **kd_tree/ball_tree** sebaiknya tidak dipakai untuk data berdimensi tinggi seperti 128D.
- ANN baru menguntungkan ketika jumlah data jauh lebih besar atau index dipakai berulang kali untuk banyak query, sehingga biaya build teramortisasi.

## 8. Catatan Metodologi

- Setiap metode dijalankan satu kali tanpa pengulangan atau pemanasan, sehingga variasi antar-run mungkin terjadi.
- Dataset berupa angka acak uniform, bukan data nyata. Pola recall dan waktu dapat berbeda pada data sebenarnya.
- Android dan Colab sama-sama memakai `float32` dan `np.random.seed(42)`.
- Waktu Android diukur dengan `time.perf_counter()`; Colab memakai `time.time()` pada notebook `Tugas.ipynb`.
