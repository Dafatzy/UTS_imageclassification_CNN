# Landscape AI Classifier — Deep Learning UTS

Project ini menghubungkan penelitian klasifikasi citra enam kelas Intel Image Classification dengan aplikasi Streamlit untuk demonstrasi inference dua model.

## Struktur proyek

```text
Landscape_AI_Classifier/
├── app.py
├── requirements.txt
├── requirements-ui-only.txt
├── requirements-models.txt
├── run_app.bat
├── assets/logo.png
├── models/                 # model hasil Google Colab diletakkan di sini
├── results/                # metrik, grafik, confusion matrix, metadata
├── notebooks/
│   └── Intel_Landscape_Classification_Tahap1_2_AUTO_DOWNLOAD.ipynb
├── guides/PANDUAN_TAHAP_1_2_3.md
└── .streamlit/config.toml
```

## Tahap 1–2: jalankan notebook

1. Buka [Google Colab](https://colab.research.google.com/).
2. Upload notebook dari folder `notebooks/` ke Colab.
3. Pilih Runtime → Change runtime type → T4 GPU jika tersedia.
4. Jalankan semua cell dari atas berurutan.
5. Gunakan file dengan nama `AUTO_DOWNLOAD` (bukan notebook lama yang meminta `kaggle.json`). Dataset diunduh otomatis dari mirror publik Hugging Face. Notebook tidak meminta `kaggle.json` dan tidak menampilkan `files.upload()` untuk dataset.
6. Izinkan akses Google Drive saat diminta untuk menyimpan checkpoint dan hasil di `MyDrive/landscape_dl_project`.
7. Jalankan eksperimen E1–E4, evaluasi validation dan test, lalu unduh `landscape_ai_classifier_artifacts.zip` dari cell terakhir.

Dataset: [Intel Image Classification di Kaggle](https://www.kaggle.com/datasets/puneet6060/intel-image-classification). Mirror otomatis: [Hugging Face dataset](https://huggingface.co/datasets/sfarrukhm/intel-image-classification). Dataset mirror memiliki folder `seg_train`, `seg_test`, serta enam kelas Buildings, Forest, Glacier, Mountain, Sea, dan Street. Hanya folder train/test berlabel yang dipakai untuk supervised learning.

## Tahap 3: hubungkan model dan jalankan aplikasi

1. Ekstrak `landscape_ai_classifier_artifacts.zip` yang diunduh dari Colab.
2. Gabungkan folder `models/` dan `results/` ke folder proyek ini.
3. Di CMD/PowerShell yang sedang berada di folder proyek, jalankan:

```powershell
py -m pip install -r requirements.txt
py -m streamlit run app.py
```

Aplikasi menerima satu gambar uji baru. Upload di aplikasi adalah **input inference**, bukan dataset training. Dengan kedua checkpoint tersedia, Custom CNN dan MobileNetV2 memproses gambar yang sama, menampilkan prediksi dan distribusi skor, lalu menyediakan laporan PDF.

Sebelum checkpoint tersedia, halaman Prediksi tetap menampilkan upload, preview, informasi preprocessing, alur arsitektur, dan penjelasan E1–E4. Aplikasi tidak membuat angka prediksi palsu.

## Arsitektur & eksperimen

- **E1:** Custom CNN tanpa augmentasi (baseline from scratch).
- **E2:** Custom CNN dengan augmentasi.
- **E3:** MobileNetV2 pretrained ImageNet, backbone dibekukan (feature extraction).
- **E4:** MobileNetV2 fine-tuning 30 layer terakhir; Batch Normalization tetap dibekukan.

Input 150×150 RGB. `seg_train` dibagi stratified 80:20 menjadi training-validation. `seg_test` dipakai sebagai holdout test. Seleksi model berdasarkan validation Macro F1; test dipakai untuk evaluasi final.

## GitHub dan deployment publik

Upload isi folder proyek ini ke repository GitHub. Pastikan `app.py` dan `requirements.txt` berada di root repository. Model `.keras` dan artefak evaluasi tidak diabaikan oleh `.gitignore` karena diperlukan aplikasi; dataset mentah dan ZIP sementara tidak diikutkan.

Setelah repository GitHub siap, buka [Streamlit Community Cloud](https://share.streamlit.io/), pilih **Create app**, repository, branch `main`, dan entrypoint `app.py`. Community Cloud akan memberi URL publik `*.streamlit.app`. Lihat [panduan deployment resmi](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy).

Sebelum mengunggah model, periksa ukuran masing-masing file. Jika satu file melebihi batas Git biasa, gunakan Git LFS atau simpan artefak model di hosting terpisah dan sesuaikan proses unduh di aplikasi.

## Status pekerjaan

Source code notebook dan aplikasi sudah disiapkan serta diperiksa sintaks/struktur. **Training aktual belum terjadi sampai notebook dijalankan di Google Colab.** Model, metrik, dan hasil prediksi hanya dianggap sah setelah proses training/evaluasi itu selesai.
