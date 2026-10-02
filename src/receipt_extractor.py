import json
import os
import re
from pathlib import Path
import lmstudio as lms

# 1. Konfigurasi Client LM Studio
lms.configure_default_client("192.168.100.232:1234")

RAW_DIR = Path("data/raw")
REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(exist_ok=True)

MODEL = os.environ.get("LM_STUDIO_MODEL", "qwen2.5-vl-7b-instruct:2")
model = lms.llm(MODEL)

# Cari semua file gambar (.png, .jpg, .jpeg) di folder data/raw
image_files = list(RAW_DIR.glob("*.png")) + list(RAW_DIR.glob("*.jpg")) + list(RAW_DIR.glob("*.jpeg"))

print(f"Ditemukan {len(image_files)} gambar nota di folder {RAW_DIR}:\n")

prompt_detail = """
Analisis dan ekstrak seluruh data dari gambar nota/struk ini secara sangat teliti.

Keluarkan HANYA JSON valid tanpa teks pengantar, deskripsi, atau format markdown block (seperti ```json).

Gunakan struktur JSON berikut:
{
  "merchant": {
    "nama": "Nama toko/merchant",
    "alamat": "Alamat toko jika ada, atau null",
    "telepon": "Nomor telepon jika ada, atau null"
  },
  "transaksi": {
    "no_nota": "Nomor nota/faktur/invoice jika ada, atau null",
    "tanggal": "Tanggal transaksi (YYYY-MM-DD atau sesuai teks)",
    "waktu": "Waktu/jam transaksi jika ada, atau null",
    "metode_pembayaran": "Cash/Debit/Kredit/Transfer dll., atau null"
  },
  "items": [
    {
      "kode_barang": "Kode barang jika ada, atau null",
      "nama_barang": "Nama/Deskripsi barang",
      "jumlah": 0,
      "satuan": "Pcs/Kg/Box dll., atau null",
      "harga_satuan": 0,
      "diskon_item": 0,
      "total_harga": 0
    }
  ],
  "ringkasan_biaya": {
    "subtotal": 0,
    "diskon_total": 0,
    "pajak": 0,
    "layanan_biaya_lain": 0,
    "grand_total": 0
  },
  "catatan": "Catatan khusus atau terbilang pada nota jika ada, atau null"
}

Aturan Ketat:
1. Periksa setiap angka dan teks dengan cermat, jangan mengarang data.
2. Jika ada diskon, pastikan tercatat di diskon_total dan grand_total.
3. Semua nilai angka/nominal harus berupa tipe data integer/float tanpa simbol mata uang.
4. Jika informasi tidak ditemukan di dalam nota, isi dengan null atau 0 untuk nilai numerik.
"""

# Loop untuk memproses setiap gambar satu per satu
for img_path in image_files:
    print(f"=== Memproses Gambar: {img_path.name} ===")
    try:
        image = lms.prepare_image(str(img_path))
        chat = lms.Chat()
        chat.add_user_message(prompt_detail, images=[image])

        prediction = model.respond(chat)
        raw_content = prediction.content.strip()

        # Pembersihan Teks Markdown Block
        cleaned_content = re.sub(r"^```(?:json)?\s*", "", raw_content, flags=re.IGNORECASE | re.MULTILINE)
        cleaned_content = re.sub(r"\s*```$", "", cleaned_content, flags=re.MULTILINE).strip()

        result = json.loads(cleaned_content)

        # Simpan hasil masing-masing nota (contoh: reports/nota-sample.json)
        output_file = REPORTS_DIR / f"{img_path.stem}.json"
        output_file.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        
        print(f"✅ Selesai! Hasil disimpan ke: {output_file}\n")

    except Exception as e:
        print(f"❌ Gagal memproses {img_path.name}: {e}\n")

print("--- SEMUA GAMBAR SELESAI DIPROSES ---")
