# Decision Tree Web App

Aplikasi web berbasis **Flask** untuk mengintegrasikan model **Decision Tree Classifier**.  
Pengguna dapat mengunggah dataset (CSV/XLSX), melatih model klasifikasi, menampilkan visualisasi pohon keputusan, dan melakukan prediksi berdasarkan input pengguna.

## 🚀 Fitur
- Upload dataset (CSV/XLSX).
- Preprocessing otomatis (One-Hot Encoding untuk variabel kategorikal).
- Split data menjadi train/test.
- Training model Decision Tree dengan scikit-learn.
- Visualisasi pohon keputusan menggunakan Graphviz.
- Menampilkan akurasi dan laporan klasifikasi.
- Prediksi berdasarkan input manual dari user.

## 🛠️ Tech Stack
- Python 3.x
- Flask
- pandas
- scikit-learn
- numpy
- Graphviz

## 📂 Struktur Proyek
decision-tree/
│── app.py # Main Flask app
│── requirements.txt # Dependencies
│── templates/
│ ├── index.html # Halaman upload
│ ├── data.html # Halaman hasil training
│ └── predict.html # Halaman prediksi
│── static/ # File CSS/JS/images


## 📦 Instalasi
1. Clone repository:
   ```bash
   git clone https://github.com/username/decision-tree-app.git
   cd decision-tree
2. Install dependencies:
   ```bash
   pip install flask pandas scikit-learn graphviz numpy openpyxl
3. Jalankan aplikasi:
   ```bash
   python app.py
4. Buka browser:
   ```bash
   http://127.0.0.1:5000
