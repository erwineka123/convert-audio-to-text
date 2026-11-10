# Audio ke Teks

Aplikasi Streamlit untuk mentranskripsikan file audio menggunakan
Faster-Whisper. Aplikasi menggunakan CPU dan model `tiny`, `base`, atau
`small`; hasil transkripsi langsung dapat diunduh sebagai PDF. Model
`large-v3` dari notebook memerlukan GPU dan tidak cocok untuk Streamlit
Community Cloud.

## Jalankan lokal

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Pada Linux/macOS, aktivasi environment virtual menggunakan
`source .venv/bin/activate`.

## Deploy ke Streamlit Community Cloud

1. Push `app.py` dan `requirements.txt` ke repository GitHub.
2. Masuk ke [Streamlit Community Cloud](https://share.streamlit.io/) dan pilih
   **Create app**.
3. Hubungkan repository, branch, dan set **Main file path** ke `app.py`.
4. Pilih **Deploy**. Streamlit akan memasang dependency dari
   `requirements.txt`.

Saat pertama dipakai, model Whisper diunduh dan disimpan dalam cache aplikasi.
Gunakan model `tiny` atau `base` dan audio pendek untuk menguji deploy CPU;
model lebih berat atau audio panjang dapat lambat atau kehabisan memori.
File dan transkrip pada aplikasi cloud tidak disimpan permanen, jadi unduh
hasil sebelum menutup sesi.
