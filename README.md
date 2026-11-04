# Image Classifier
**KELOMPOK 8:**
Dafa Aditya Nazwan Priyadi – 231111548
M. Sendy Aulia – 231111455
M. Alief Mufthi – 231111319
Rasidin Harahap – 231110869
 Ade Kurnia Setialu - 231110333


## 1. Project Overview

Sistem melakukan klasifikasi gambar ke dalam enam kategori:

- Buildings
- Forest
- Glacier
- Mountain
- Sea
- Street

**Model yang digunakan:**

| Model | Metode |
|---|---|
| Custom CNN | Training from scratch |
| MobileNetV2 | Transfer learning dan fine-tuning |

## 2. Dataset

Dataset yang digunakan adalah **Intel Image Classification Dataset**.

- [Dataset Intel Image Classification – Kaggle](https://www.kaggle.com/datasets/puneet6060/intel-image-classification)
- [Dataset Mirror – Hugging Face](https://huggingface.co/datasets/sfarrukhm/intel-image-classification)

Dataset memiliki enam kelas gambar. Data training digunakan untuk proses pelatihan dan validasi, sedangkan data testing digunakan sebagai data evaluasi akhir.

Gambar prediction tanpa label tidak digunakan dalam evaluasi supervised learning.

## 3. Model Architecture and Experiments

Penelitian membandingkan empat skenario eksperimen:

| Eksperimen | Model | Konfigurasi |
|---|---|---|
| E1 | Custom CNN | Tanpa augmentasi |
| E2 | Custom CNN | Dengan augmentasi |
| E3 | MobileNetV2 | Feature extraction |
| E4 | MobileNetV2 | Fine-tuning 30 layer terakhir |

Seleksi model terbaik dilakukan berdasarkan Validation Macro F1. Data testing digunakan untuk evaluasi akhir.

**Konfigurasi utama:**
- Input image: 150 × 150 pixels
- Color mode: RGB
- Jumlah kelas: 6
- Optimizer: Adam
- Loss function: Sparse Categorical Crossentropy
- Evaluasi: Accuracy, Precision, Recall, Macro F1-score, dan Confusion Matrix.

## 4. Project Structure

```text
imageclassification/
│
├── .streamlit/
│   └── config.toml
│
├── assets/
│   └── logo.png
│
├── guides/
│   └── PANDUAN_TAHAP_1_2_3.md
│
├── models/
│   ├── best_overall_model.keras
│   ├── custom_cnn_best.keras
│   └── mobilenetv2_best.keras
│
├── notebooks/
│   └── Kelompok8_Image_Classification.ipynb
│
├── results/
│   ├── class_names.json
│   ├── dataset_class_summary.csv
│   ├── experiment_summary.json
│   ├── model_selection.json
│   ├── validation_metrics.csv
│   ├── test_metrics_all_experiments.csv
│   ├── training_curves_all_experiments.png
│   ├── confusion_matrix_best_model.png
│   ├── misclassified_examples.png
│   └── training history E1–E4
│
├── app.py
├── requirements.txt
├── requirements-models.txt
├── requirements-ui-only.txt
├── run_app.bat
├── README.md
└── .gitignore
```

## 5. Model Evaluation

Evaluasi dilakukan terhadap empat eksperimen E1–E4 menggunakan data testing yang sama.

Hasil evaluasi disimpan dalam folder `results/`, meliputi:

- Accuracy, Precision, Recall, dan Macro F1-score.
- Confusion Matrix.
- Training dan Validation Curves.
- Riwayat training setiap eksperimen.
- Analisis kesalahan klasifikasi.
- Ringkasan pemilihan model terbaik.

Nilai metrik akhir dapat dilihat pada file `test_metrics_all_experiments.csv` dan `experiment_summary.json`.

## 6. Application Features

Aplikasi web dikembangkan menggunakan Streamlit dengan fitur:

1. Splash screen dan onboarding.
2. Dashboard pengenalan proyek.
3. Upload gambar untuk klasifikasi.
4. Prediksi otomatis menggunakan Custom CNN dan MobileNetV2.
5. Perbandingan hasil prediksi kedua model.
6. Visualisasi probabilitas enam kelas.
7. Informasi arsitektur dan eksperimen Deep Learning.
8. Halaman evaluasi performa model.
9. Download laporan hasil prediksi dalam format PDF.

Aplikasi menggunakan checkpoint model hasil training Google Colab. Prediksi hanya dilakukan menggunakan model yang telah dilatih.

## 7. Installation and Usage

### Requirements

- Python 3.13 (atau versi yang sesuai dengan dependensi).
- pip.
- Git.
- Model checkpoint hasil training.

### Install Dependencies

Clone repository:

```bash
git clone https://github.com/USERNAME/imageclassification.git
```

Masuk ke folder proyek:

```bash
cd imageclassification
```

Install dependensi aplikasi:

```bash
py -m pip install -r requirements.txt
```

Install dependensi model untuk menjalankan inference:

```bash
py -m pip install -r requirements-models.txt
```

### Run Application

Jalankan aplikasi menggunakan:

```bash
py -m streamlit run app.py
```

Atau pada Windows, gunakan:

```text
run_app.bat
```

Aplikasi akan berjalan secara lokal melalui browser.

## 8. Deployment

Aplikasi dapat dideploy menggunakan Streamlit Community Cloud.

1. Siapkan repository GitHub.
2. Buka [Streamlit Community Cloud](https://share.streamlit.io/).
3. Hubungkan akun GitHub.
4. Pilih repository `imageclassification`.
5. Pilih branch `main`.
6. Tentukan file utama `app.py`.
7. Selesaikan proses deployment.

Dokumentasi resmi: [Streamlit Deployment Guide](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy).

## 9. Author and Project Information

**Project:** Landscape AI Classifier  
**Subject:** Deep Learning  
**Project Type:** UTS  
**Dataset:** Intel Image Classification  
**Framework:** TensorFlow / Keras  
**Web Framework:** Streamlit  
**Repository:** `imageclassification`

---

*Landscape AI Classifier — Deep Learning Project.*
