import sys
import json
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import pandas as pd


# ============================================================
# KONFIGURASI PATH
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(BASE_DIR, "data", "tickets.csv")
MODEL_PATH = os.path.join(BASE_DIR, "model", "model.joblib")
METRICS_PATH = os.path.join(BASE_DIR, "reports", "metrics.json")


# ============================================================
# MEMBUAT FOLDER JIKA BELUM ADA
# ============================================================

os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
os.makedirs(os.path.dirname(METRICS_PATH), exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

def load_data():
    """
    Membaca dataset tiket dari file CSV.
    """

    if not os.path.exists(DATA_PATH):
        print(f"ERROR: File dataset tidak ditemukan:")
        print(DATA_PATH)
        sys.exit(1)

    df = pd.read_csv(DATA_PATH)

    # Validasi kolom
    required_columns = ["ticket_id", "text", "channel", "urgent"]

    for column in required_columns:
        if column not in df.columns:
            print(f"ERROR: Kolom '{column}' tidak ditemukan di dataset.")
            sys.exit(1)

    # Hapus data kosong
    df = df.dropna(subset=["text", "urgent"])

    return df


# ============================================================
# TRAINING
# ============================================================

def train():
    """
    Melatih model menggunakan:
    TF-IDF + Logistic Regression

    TF-IDF dan Logistic Regression digabung dalam satu Pipeline
    sehingga preprocessing dan classifier tersimpan sebagai satu
    artefak model.
    """

    print("=" * 60)
    print("TRAINING MODEL")
    print("=" * 60)

    # Load dataset
    df = load_data()

    print(f"Jumlah data : {len(df)}")
    print()

    # Input
    X = df["text"].astype(str)

    # Target
    y = df["urgent"].astype(int)

    # Split data train dan test
    #
    # stratify digunakan agar distribusi label 0 dan 1 tetap
    # relatif sama antara train dan test.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y
    )

    print(f"Data training : {len(X_train)}")
    print(f"Data testing  : {len(X_test)}")
    print()

    # ========================================================
    # PIPELINE
    # ========================================================

    pipeline = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                lowercase=True,
                ngram_range=(1, 2),
                max_features=5000
            )
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                random_state=42
            )
        )
    ])

    # Training
    print("Memulai training...")

    pipeline.fit(X_train, y_train)

    print("Training selesai.")
    print()

    # ========================================================
    # EVALUASI
    # ========================================================

    y_pred = pipeline.predict(X_test)

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    metrics = {
        "model": "TF-IDF + Logistic Regression",
        "dataset": "data/tickets.csv",
        "total_data": len(df),
        "training_data": len(X_train),
        "testing_data": len(X_test),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1_score": round(float(f1), 4),
        "accuracy": round(float(accuracy), 4)
    }

    # ========================================================
    # SIMPAN MODEL
    # ========================================================

    joblib.dump(
        pipeline,
        MODEL_PATH
    )

    print(f"Model disimpan:")
    print(MODEL_PATH)
    print()

    # ========================================================
    # SIMPAN METRICS
    # ========================================================

    with open(
        METRICS_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            metrics,
            file,
            indent=4
        )

    print(f"Metrics disimpan:")
    print(METRICS_PATH)
    print()

    # ========================================================
    # HASIL EVALUASI
    # ========================================================

    print("=" * 60)
    print("HASIL EVALUASI")
    print("=" * 60)

    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")

    print("=" * 60)


# ============================================================
# INFERENCE / PREDICTION
# ============================================================

def predict(text):
    """
    Melakukan prediksi terhadap tiket baru menggunakan model
    yang sudah disimpan.
    """

    print("=" * 60)
    print("INFERENCE")
    print("=" * 60)

    # Cek apakah model tersedia
    if not os.path.exists(MODEL_PATH):
        print("ERROR: Model belum tersedia.")
        print()
        print("Silakan training terlebih dahulu:")
        print()
        print("python src/pipeline.py train")
        sys.exit(1)

    # Load model
    pipeline = joblib.load(MODEL_PATH)

    # Prediksi label
    prediction = pipeline.predict([text])[0]

    # Probabilitas
    probability = pipeline.predict_proba([text])[0]

    # Probabilitas untuk label 1
    urgent_probability = probability[1]

    # Konversi label ke teks
    if prediction == 1:
        label = "URGENT"
    else:
        label = "NOT URGENT"

    print(f"Tiket     : {text}")
    print()
    print(f"Prediksi  : {label}")
    print(f"Label     : {prediction}")
    print(f"Probabilitas urgent : {urgent_probability:.4f}")

    print("=" * 60)


# ============================================================
# MAIN
# ============================================================

def main():

    if len(sys.argv) < 2:
        print("Penggunaan:")
        print()
        print("Training:")
        print("python src/pipeline.py train")
        print()
        print("Prediction:")
        print('python src/pipeline.py predict "Teks tiket baru"')
        sys.exit(1)

    command = sys.argv[1].lower()

    # TRAIN
    if command == "train":
        train()

    # PREDICT
    elif command == "predict":

        if len(sys.argv) < 3:
            print("ERROR: Masukkan teks tiket.")
            print()
            print('Contoh:')
            print(
                'python src/pipeline.py predict '
                '"Dana tertagih dua kali dan transaksi saya gagal"'
            )
            sys.exit(1)

        text = " ".join(sys.argv[2:])

        predict(text)

    else:
        print(f"ERROR: Command '{command}' tidak dikenal.")
        print()
        print("Gunakan:")
        print("python src/pipeline.py train")
        print(
            'python src/pipeline.py predict "Teks tiket baru"'
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
