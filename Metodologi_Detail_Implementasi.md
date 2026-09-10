# Metodologi Detail Implementasi
## Penelitian: ARIMA vs LSTM vs Hybrid ARIMA-LSTM
### Forecasting Harga Saham Sektor Batubara IDX (ADRO, PTBA, INDY, ITMG)

---

## 1. Sumber Dataset

### 1.1 Asal dan Cara Akuisisi Data

Data diperoleh secara otomatis dari **Yahoo Finance** menggunakan library Python `yfinance`, yang menyediakan akses terprogram terhadap data pasar saham historis. Proses akuisisi dilakukan menggunakan kode:

```python
import yfinance as yf
ticker_list = ["ADRO.JK", "PTBA.JK", "INDY.JK", "ITMG.JK"]
df = yf.download(ticker_list, start="2004-01-01", end="2024-12-31")
```

Suffix `.JK` menunjukkan saham terdaftar di **Bursa Efek Indonesia (BEI/IDX)**.

### 1.2 Spesifikasi Dataset

| Properti | Detail |
|---|---|
| **Emiten** | ADRO, PTBA, INDY, ITMG |
| **Rentang Waktu** | 2 Januari 2004 – 31 Desember 2024 |
| **Frekuensi** | Harian (*business days*, Senin-Jumat) |
| **Ukuran Perkiraan** | +/- 5.199 observasi per emiten |
| **File yang disimpan** | `data/harga_saham_yfinance.csv` |

### 1.3 Atribut Data yang Tersedia

Setiap ticker pada Yahoo Finance menyediakan kolom: `Open`, `High`, `Low`, `Close`, `Adj Close`, `Volume`. Dari seluruh atribut tersebut, **hanya kolom `Adj Close` (Adjusted Closing Price)** yang digunakan untuk pemodelan.

---

## 2. Alasan Hanya Menggunakan Closing Price (Adj Close)

Penelitian ini secara eksplisit **hanya menggunakan `Adjusted Closing Price`** sebagai satu-satunya fitur input model. Terdapat 5 justifikasi metodologis yang mendasari keputusan ini:

### 2.1 Representasi Harga yang Paling Informatif
*Closing price* merepresentasikan **konsensus akhir pasar** di setiap hari perdagangan. Harga ini mencerminkan seluruh informasi yang tersedia (berita, sentimen investor, aksi korporasi) yang telah terserap dan dicerminkan oleh pasar sepanjang hari trading.

### 2.2 Penyesuaian (Adjustment) untuk Aksi Korporasi
Kolom `Adj Close` berbeda dari `Close` biasa karena telah **disesuaikan secara retroaktif** terhadap:
- **Stock Split** (pemecahan saham): harga disesuaikan agar time-series tetap kontinu.
- **Dividen tunai**: harga disesuaikan agar tidak terjadi lompatan artifisial pada ex-dividend date.
- **Rights Issue**: harga disesuaikan terhadap penerbitan saham baru.

Tanpa penyesuaian ini, time-series harga akan memiliki *structural breaks* (diskontinuitas) yang palsu dan menyesatkan proses pelatihan model.

### 2.3 Landasan pada Teori Keuangan Standar
Penggunaan closing price adalah **standar de facto** dalam penelitian keuangan kuantitatif dan time-series econometrics. Hampir seluruh literatur prediksi harga saham menggunakan univariate closing price sebagai variabel target untuk:
- Menjamin komparabilitas dengan penelitian lain.
- Menghindari kompleksitas multivariate yang tidak perlu pada tahap ini.
- Menjaga model tetap *parsimonious* (sederhana namun akurat).

### 2.4 Konsistensi dengan Asumsi Model ARIMA
Model **ARIMA adalah model univariat** (single time-series). Memasukkan atribut lain (Open, High, Low, Volume) akan mengubahnya menjadi model ARIMAX (ARIMA with Exogenous Variables), yang di luar cakupan penelitian komparatif ini.

### 2.5 Fokus Penelitian: Perbandingan Murni Antar Model
Karena tujuan utama penelitian adalah **membandingkan ARIMA vs LSTM vs Hybrid ARIMA-LSTM** secara setara, maka ketiga model harus dilatih pada input yang identik. Penggunaan fitur tunggal (`Adj Close`) memastikan perbedaan kinerja antar model murni berasal dari **perbedaan arsitektur model**, bukan dari perbedaan fitur input.

---

## 3. Preprocessing Data

Preprocessing dilakukan dalam 4 tahap berurutan.

### 3.1 Pemuatan Data dan Penanganan Missing Value

```python
df = pd.read_csv(
    "data/harga_saham_yfinance.csv",
    index_col=0,
    parse_dates=True
)
```

Tanggal libur bursa menyebabkan *gap* pada index. Penanganan dilakukan dengan:

```python
df = df.ffill().bfill()
```

- **`ffill()` (Forward Fill):** Mengisi hari libur dengan harga terakhir yang valid sebelumnya.
- **`bfill()` (Backward Fill):** Menangani kasus jika data awal time-series kosong.

### 3.2 Uji Stasioneritas — Augmented Dickey-Fuller (ADF) Test

Stasioneritas adalah **syarat wajib** model ARIMA. Pengujian dilakukan menggunakan ADF Test:

```python
from statsmodels.tsa.stattools import adfuller
result = adfuller(series, autolag='AIC')
# H0: unit root ada (tidak stasioner), ditolak jika p-value < 0.05
```

Seluruh keempat emiten (ADRO, PTBA, INDY, ITMG) ditemukan **tidak stasioner pada level (d=0)**, sehingga diperlukan *first-order differencing* (d=1). Hal ini dikonfirmasi dari nilai ARIMA optimal yang semuanya menggunakan d=1.

### 3.3 Min-Max Scaling untuk LSTM

Data harga mentah (dalam satuan Rupiah) memiliki rentang nilai yang sangat berbeda antar emiten (misalnya ITMG +/- 30.000 IDR vs INDY +/- 2.000 IDR). Normalisasi diperlukan agar proses konvergensi LSTM lebih stabil:

```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(-1, 1))  # bukan (0,1)!
train_sc = scaler.fit_transform(series_train.values.reshape(-1, 1)).flatten()
test_sc  = scaler.transform(series_test.values.reshape(-1, 1)).flatten()
```

> **PENTING:** `scaler.fit_transform()` hanya dilakukan pada **training data**. Kemudian `scaler.transform()` (tanpa `fit`) diterapkan pada test data, untuk mencegah *data leakage* dari informasi masa depan ke dalam proses pelatihan.

**Rentang scaling:** `feature_range=(-1, 1)` dipilih (bukan `(0, 1)`) karena harga saham dapat bergerak naik maupun turun, dan rentang simetris ini lebih sesuai dengan fungsi aktivasi `tanh` yang digunakan secara implisit dalam sel LSTM.

### 3.4 Pembentukan Sliding Window Sequence

Data 1D yang terskalakan diubah menjadi pasangan **(X, y)** untuk pembelajaran terawasi:

```python
def buat_sequences(data: np.ndarray, window: int):
    X, y = [], []
    for i in range(window, len(data)):
        X.append(data[i-window:i])  # input: window hari sebelumnya
        y.append(data[i])           # target: hari ke-i
    return np.array(X)[..., np.newaxis], np.array(y)
    # X shape: (n_samples, window_size, 1)
    # y shape: (n_samples,)
```

- **Window Size:** `WINDOW = 30` (30 hari kalender kerja, sekitar 1 bulan).
- **Implikasi:** 30 hari pertama pada *test set* tidak dapat menjadi target prediksi (karena membutuhkan lookback), sehingga evaluasi dilakukan pada `df_test.values[WINDOW:]`.

---

## 4. Train / Validation / Test Split

### 4.1 Pembagian Train dan Test

Pembagian dilakukan berdasarkan **tanggal kalender tetap** (*hard cutoff*), bukan secara persentase:

```python
CUTOFF = "2022-12-31"

df_train = df[df.index <= CUTOFF].copy()  # Data 2004 - 2022
df_test  = df[df.index >  CUTOFF].copy()  # Data 2023 - 2024
```

| Set | Rentang Waktu | Hari Kerja (Perkiraan) | Proporsi |
|---|---|---|---|
| **Train** | 2004-01-01 - 2022-12-31 | +/- 4.696 hari | ~90.3% |
| **Test** | 2023-01-01 - 2024-12-31 | +/- 503 hari | ~9.7% |

**Alasan Cutoff 2022:**
- Train set mencakup > 18 tahun data historis untuk menangkap pola jangka panjang.
- Test set mencakup 2 tahun (2023-2024) sebagai *out-of-sample evaluation*, mewakili periode pasca-pandemi yang bersifat dinamis.

### 4.2 Pembagian Train dan Validation (Internal LSTM)

Dari training set (2004-2022), sebagian digunakan sebagai **validation set internal** selama pelatihan LSTM:

```python
mdl.fit(
    X_tr, y_tr,
    validation_split=0.15,  # 15% akhir dari training data -> validation
    ...
)
```

- **15% dari training set** (~704 hari terakhir dari 2004-2022) -> validation.
- **85% sisanya** (~3.992 hari) -> training aktual.
- Validation digunakan oleh `EarlyStopping` untuk menentukan kapan training berhenti.

> **Catatan:** `validation_split` di Keras mengambil bagian **akhir** (bukan acak) dari data training. Hal ini tepat untuk time-series karena menghindari *data leakage* dari masa depan.

---

## 5. Pemilihan Parameter ARIMA

### 5.1 Metode Pemilihan

Parameter ARIMA (p, d, q) ditentukan menggunakan **Grid Search berbasis AIC (Akaike Information Criterion)**:

```python
from statsmodels.tsa.statespace.sarimax import SARIMAX

best_aic = np.inf
best_order = None
for p in range(0, 5):
    for q in range(0, 5):
        try:
            mdl = SARIMAX(series, order=(p, 1, q), trend="n",
                          enforce_stationarity=False,
                          enforce_invertibility=False)
            res = mdl.fit(disp=False)
            if res.aic < best_aic:
                best_aic, best_order = res.aic, (p, 1, q)
        except:
            continue
```

- d=1 ditetapkan secara *a priori* berdasarkan hasil ADF Test.
- p dan q di-*grid search* pada rentang [0, 4] (25 kombinasi per emiten).
- Model dengan **AIC terkecil** dipilih.

### 5.2 Hasil Pemilihan Parameter Terbaik

| Emiten | Orde ARIMA Terbaik | Keterangan |
|---|---|---|
| **ADRO** | ARIMA(3, 1, 3) | AR lag 3, differencing 1, MA lag 3 |
| **PTBA** | ARIMA(1, 1, 3) | AR lag 1, differencing 1, MA lag 3 |
| **INDY** | ARIMA(1, 1, 3) | AR lag 1, differencing 1, MA lag 3 |
| **ITMG** | ARIMA(3, 1, 3) | AR lag 3, differencing 1, MA lag 3 |

Parameter ini disimpan ke dalam file `data/arima_best_orders.pkl` untuk digunakan kembali di tahap Hybrid ARIMA-LSTM.

---

## 6. Arsitektur LSTM

### 6.1 Fungsi Pembangunan Model

Arsitektur LSTM diimplementasikan dalam fungsi `bangun_lstm()` yang dapat dipakai ulang (*reusable*) untuk model LSTM Murni maupun LSTM pada residual ARIMA (Hybrid):

```python
def bangun_lstm(window: int,
                lstm_units: int = 64,
                dropout: float  = 0.2,
                lr: float       = 1e-3) -> tf.keras.Model:

    model = Sequential([
        Input(shape=(window, 1)),

        LSTM(lstm_units, return_sequences=True,
             kernel_regularizer=l2(1e-4)),      # Layer LSTM ke-1
        Dropout(dropout),                        # Regularisasi

        LSTM(lstm_units, return_sequences=False,
             kernel_regularizer=l2(1e-4)),       # Layer LSTM ke-2
        Dropout(dropout),                        # Regularisasi

        Dense(32, activation="relu"),            # Lapisan fully-connected
        Dense(1)                                 # Output: 1 nilai prediksi
    ])

    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss="mse",
        metrics=["mae"]
    )
    return model
```

### 6.2 Detail Komponen Arsitektur

| Lapisan | Konfigurasi | Fungsi |
|---|---|---|
| **Input** | Shape (30, 1) | Window 30 hari x 1 fitur |
| **LSTM Layer 1** | 64 units, return_sequences=True, L2 reg 1e-4 | Mengekstrak pola temporal, output ke layer berikutnya |
| **Dropout 1** | Rate 0.2 | Mencegah *overfitting* (20% neuron di-drop) |
| **LSTM Layer 2** | 64 units, return_sequences=False, L2 reg 1e-4 | Kompresi representasi ke dalam 1 vektor |
| **Dropout 2** | Rate 0.2 | Regularisasi lanjutan |
| **Dense 32** | 32 unit, aktivasi ReLU | Lapisan *fully-connected* untuk non-linearitas akhir |
| **Dense 1** | 1 unit, no aktivasi (linear) | Output tunggal: prediksi harga/residual |

---

## 7. Hyperparameter

Semua hyperparameter didefinisikan sebagai konstanta global dan diterapkan secara konsisten untuk seluruh 4 emiten dan 2 variasi model (LSTM Murni & LSTM Residual):

```python
# Cell 11 — model_ann_lstm.ipynb
WINDOW        = 30       # Panjang lookback window (hari)
LSTM_UNITS    = 64       # Jumlah neuron per layer LSTM
DROPOUT_RATE  = 0.2      # Dropout probability
LEARNING_RATE = 1e-3     # Adam initial learning rate (= 0.001)
BATCH_SIZE    = 32       # Jumlah sampel per gradient update
EPOCHS        = 100      # Maksimum epoch (EarlyStopping biasanya berhenti lebih awal)
PATIENCE      = 15       # EarlyStopping patience
```

### Justifikasi Setiap Hyperparameter

| Hyperparameter | Nilai | Alasan |
|---|---|---|
| `WINDOW = 30` | 30 hari | ~1 bulan kalender kerja; menangkap siklus *monthly* yang umum pada pasar saham |
| `LSTM_UNITS = 64` | 64 | Kapasitas cukup untuk pola time-series keuangan tanpa *over-parameterized* |
| `DROPOUT_RATE = 0.2` | 0.2 | Nilai standar, mencegah *overfitting* pada dataset ~4.700 sampel |
| `LEARNING_RATE = 1e-3` | 0.001 | Default Adam; `ReduceLROnPlateau` akan menurunkannya secara adaptif |
| `BATCH_SIZE = 32` | 32 | Kompromi antara kecepatan training dan stabilitas gradient |
| `EPOCHS = 100` | 100 | Upper bound; EarlyStopping aktif jauh sebelum epoch ke-100 |
| `PATIENCE = 15` | 15 | Toleransi 15 epoch tanpa perbaikan sebelum training dihentikan |

---

## 8. Prosedur Training

### 8.1 Callbacks yang Digunakan

Dua callback Keras digunakan untuk mengoptimalkan proses training secara otomatis:

```python
callbacks_def = [
    EarlyStopping(
        monitor="val_loss",
        patience=PATIENCE,           # Berhenti setelah 15 epoch tanpa perbaikan
        restore_best_weights=True,   # Kembalikan bobot ke epoch terbaik
        verbose=0
    ),
    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,                  # Kurangi LR menjadi 50% dari nilai saat ini
        patience=7,                  # Turunkan LR setelah 7 epoch stagnan
        min_lr=1e-6,                 # Batas minimum learning rate
        verbose=0
    ),
]
```

### 8.2 Alur Training Penuh

```
Untuk setiap emiten in [ADRO, PTBA, INDY, ITMG]:
    1. Siapkan data: scale -> buat sequences -> split train/val
    2. Build model: bangun_lstm(WINDOW, LSTM_UNITS, DROPOUT_RATE, LEARNING_RATE)
    3. Fit model:
       - Train pada 85% training data
       - Validasi pada 15% akhir training data
       - Gunakan EarlyStopping + ReduceLROnPlateau
    4. Prediksi test set:
       - pred_sc = model.predict(X_test)         # hasil dalam skala (-1, 1)
       - pred = scaler.inverse_transform(pred_sc) # kembalikan ke IDR asli
    5. Hitung metrik: MAE, RMSE, MAPE (%)
```

### 8.3 Loss Function dan Optimizer

- **Loss function:** `MSE` (Mean Squared Error) — standar untuk regresi, menghukum error besar lebih berat.
- **Optimizer:** `Adam` (Adaptive Moment Estimation) — optimizer adaptif yang umumnya digunakan untuk LSTM karena penanganan *sparse gradient* yang baik.

### 8.4 Evaluasi Final (Inverse Transform)

Karena training dilakukan pada data yang di-*scale* ke `(-1, 1)`, prediksi harus dikembalikan ke skala asli (IDR) sebelum dihitung metriknya:

```python
pred_sc = mdl.predict(d["X_te"], verbose=0).flatten()
pred    = d["scaler"].inverse_transform(pred_sc.reshape(-1,1)).flatten()

# Aktual yang digunakan untuk evaluasi (dipotong WINDOW hari pertama):
aktual  = df_test[ticker].dropna().values[WINDOW:]
```

### 8.5 Formula Metrik Evaluasi

**MAE:**
```
MAE = (1/n) * sum(|y_i - y_hat_i|)
```

**RMSE:**
```
RMSE = sqrt((1/n) * sum((y_i - y_hat_i)^2))
```

**MAPE (%):**
```
MAPE = (100/n) * sum(|y_i - y_hat_i| / y_i)
```

---

## 9. Implementasi agar Reproducible

Reproducibility dijamin melalui 4 mekanisme yang diimplementasikan di **Cell 3** notebook setiap kali dijalankan:

### 9.1 Pengaturan Random Seed

```python
SEED = 2024
np.random.seed(SEED)       # Seed untuk NumPy
tf.random.set_seed(SEED)   # Seed untuk TensorFlow/Keras
```

Nilai `SEED = 2024` dipilih untuk menandai tahun akhir data historis (2024).

### 9.2 Penekanan Variabilitas

```python
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Suppress TensorFlow log
```

### 9.3 Penyimpanan Model Terlatih

```python
for ticker in ticker_list:
    lstm_models_pure[ticker].save(f"models/lstm_pure_{ticker}.keras")
    lstm_models_resid[ticker].save(f"models/lstm_resid_{ticker}.keras")
```

Dengan menyimpan model, eksperimen lanjutan dapat dilakukan **tanpa melatih ulang dari awal**, menjamin hasil yang identik.

### 9.4 Penyimpanan Scaler

```python
with open("scalers_lstm.pkl", "wb") as f:
    pickle.dump({t: data_lstm[t]["scaler"] for t in ticker_list}, f)
```

Objek `MinMaxScaler` disimpan menggunakan `pickle`, sehingga proses *inverse transform* di masa mendatang menggunakan parameter `min` dan `max` yang sama persis.

### 9.5 Versi Library yang Digunakan

| Library | Versi | Fungsi |
|---|---|---|
| Python | 3.11.9 | Interpreter |
| TensorFlow / Keras | 2.x (latest compatible) | Model LSTM |
| NumPy | >= 1.23 | Array computation |
| Pandas | >= 2.0 | Data manipulation |
| scikit-learn | >= 1.3 | MinMaxScaler, metrik evaluasi |
| statsmodels | >= 0.14 | ARIMA, ADF Test |
| Matplotlib | >= 3.7 | Visualisasi |
| yfinance | >= 0.2 | Akuisisi data Yahoo Finance |

---

## 10. Ringkasan Konfigurasi Eksperimen

| Item | Nilai / File |
|---|---|
| Data source | `data/harga_saham_yfinance.csv` |
| Target variable | `Adj Close` (harian, dalam IDR) |
| Training period | 2004-01-01 - 2022-12-31 |
| Test period | 2023-01-01 - 2024-12-31 |
| Validation split | 15% dari training (internal Keras) |
| Window size | 30 hari |
| Scaler | MinMaxScaler(feature_range=(-1, 1)) |
| LSTM layer | 2x LSTM(64) + 2x Dropout(0.2) + Dense(32) + Dense(1) |
| Regularisasi | L2 (1e-4) + Dropout (0.2) |
| Optimizer | Adam, lr=0.001 |
| Loss | MSE |
| Max epochs | 100 |
| EarlyStopping patience | 15 (monitor: val_loss) |
| ReduceLROnPlateau | factor=0.5, patience=7, min_lr=1e-6 |
| Random seed | 2024 |
| Model tersimpan | `models/lstm_pure_{ADRO,PTBA,INDY,ITMG}.keras` |
| Scaler tersimpan | `scalers_lstm.pkl` |
| Tabel evaluasi | `tabel05_perbandingan_model.csv`, `hasil_evaluasi_LSTM.csv` |

---

*Dokumen ini bersumber langsung dari implementasi aktual di `model_ann_lstm.ipynb`.*
