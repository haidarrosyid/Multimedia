import pandas as pd
import numpy as np
from tabulate import tabulate

# ==============================
# KONFIGURASI
# ==============================

DATA_PATH = "Smartphones_cleaned_dataset.csv"

# faktor konversi: angka di dataset × 100 = Rupiah asli
PRICE_FACTOR = 100

# kolom kriteria yang dipakai dalam SPK
CRITERIA = [
    "price",                # biaya (cost)
    "rating",               # benefit
    "battery_capacity",     # benefit
    "ram_capacity",         # benefit
    "internal_memory",      # benefit
    "primary_camera_rear",  # benefit
    "primary_camera_front", # benefit
    "screen_size",          # benefit
    "refresh_rate",         # benefit
    "has_5g",               # benefit (0/1)
    "has_nfc",              # benefit (0/1)
    "fast_charging_available"  # benefit (0/1)
]

# bobot masing-masing kriteria (silakan sesuaikan)
WEIGHTS = [
    4,   # price  (penting tapi bukan segalanya)
    5,   # rating (sangat penting)
    4,   # battery_capacity
    3,   # ram_capacity
    2,   # internal_memory
    3,   # primary_camera_rear
    2,   # primary_camera_front
    1.5, # screen_size
    1.5, # refresh_rate
    2,   # has_5g
    1,   # has_nfc
    1.5  # fast_charging_available
]

# '+' = benefit, '-' = cost
IMPACTS = [
    '-', # price
    '+', # rating
    '+', # battery_capacity
    '+', # ram_capacity
    '+', # internal_memory
    '+', # primary_camera_rear
    '+', # primary_camera_front
    '+', # screen_size
    '+', # refresh_rate
    '+', # has_5g
    '+', # has_nfc
    '+'  # fast_charging_available
]



# ==============================
# FUNGSI UTAMA TOPSIS
# ==============================


def format_currency(value: float, *, dataset_scale: bool = True) -> str:
    """Format angka menjadi string Rupiah dengan titik pemisah ribuan."""
    amount = value * PRICE_FACTOR if dataset_scale else value
    return "Rp " + f"{int(amount):,}".replace(",", ".")


def load_data(path: str) -> pd.DataFrame:
    """Load dataset smartphone dan bersihkan baris yang ada NaN di kriteria."""
    df = pd.read_csv(path)
    cols = ["brand_name", "model"] + CRITERIA
    df = df[cols].dropna().copy()
    # pastikan price numerik (masih skala asli dari dataset, misal 129900)
    df["price"] = df["price"].astype(float)
    return df


def topsis_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Hitung skor TOPSIS untuk setiap HP."""
    X = df[CRITERIA].astype(float).values

    # normalisasi vektor
    norm = np.linalg.norm(X, axis=0)
    X_norm = X / norm

    # normalisasi bobot
    w = np.array(WEIGHTS, dtype=float)
    w = w / w.sum()

    # matriks ternormalisasi berbobot
    Xw = X_norm * w

    # solusi ideal
    ideal_best = np.zeros(len(CRITERIA))
    ideal_worst = np.zeros(len(CRITERIA))

    for j, impact in enumerate(IMPACTS):
        if impact == '+':  # benefit
            ideal_best[j] = Xw[:, j].max()
            ideal_worst[j] = Xw[:, j].min()
        else:              # cost
            ideal_best[j] = Xw[:, j].min()
            ideal_worst[j] = Xw[:, j].max()

    # jarak ke solusi ideal
    dist_best = np.linalg.norm(Xw - ideal_best, axis=1)
    dist_worst = np.linalg.norm(Xw - ideal_worst, axis=1)

    scores = dist_worst / (dist_best + dist_worst)

    df_out = df.copy()
    df_out["topsis_score"] = scores
    return df_out


def recommend_smartphones(
    df: pd.DataFrame,
    budget: int | None = None,
    top_n: int = 10
) -> pd.DataFrame:
    """
    Rekomendasi smartphone berdasarkan skor TOPSIS.
    - budget: Rupiah (misal 15000000 untuk 15 juta).
    """
    df_scored = topsis_scores(df)

    # filter berdasarkan budget dalam Rupiah
    if budget is not None:
        df_scored = df_scored[df_scored["price"] * PRICE_FACTOR <= budget].copy()

    if df_scored.empty:
        print("Tidak ada HP yang memenuhi kriteria (cek lagi budget / filter).")
        return df_scored

    df_scored = df_scored.sort_values("topsis_score", ascending=False)

    result = df_scored[[
        "brand_name", "model", "price", "rating", "battery_capacity",
        "ram_capacity", "internal_memory", "primary_camera_rear",
        "primary_camera_front", "screen_size", "refresh_rate",
        "has_5g", "has_nfc", "fast_charging_available", "topsis_score"
    ]].head(top_n)

    return result


def format_result(df: pd.DataFrame) -> pd.DataFrame:
    """Format harga & skor untuk ditampilkan."""
    out = df.copy()
    # konversi ke Rupiah asli: price_dataset × PRICE_FACTOR
    out["price"] = out["price"].apply(format_currency)
    out["topsis_score"] = out["topsis_score"].round(4)
    return out


# ==============================
# "APLIKASI" SEDERHANA VIA CLI
# ==============================

def main():
    print("=== SISTEM PENDUKUNG KEPUTUSAN PEMILIHAN HP TERBAIK (TOPSIS) ===")
    print(f"Memuat data dari: {DATA_PATH}")
    df = load_data(DATA_PATH)
    print(f"Jumlah HP dalam dataset (setelah dibersihkan): {len(df)}\n")

    while True:
        print("Menu:")
        print("1. Tampilkan 10 HP terbaik (tanpa batasan budget)")
        print("2. Tampilkan rekomendasi HP berdasarkan budget maksimum")
        print("3. Keluar")

        choice = input("Pilihan [1/2/3]: ").strip()

        if choice == "1":
            result = recommend_smartphones(df, budget=None, top_n=10)
            if result.empty:
                continue
            print("\n=== 10 HP TERBAIK VERSI TOPSIS ===")
            print(tabulate(format_result(result), headers="keys", tablefmt="fancy_grid", showindex=False))
            print()

        elif choice == "2":
            try:
                budget = int(
                    input(
                        "Masukkan budget maksimum (dalam Rupiah, misal 15000000): "
                    )
                )
            except ValueError:
                print("Input budget harus berupa angka.\n")
                continue

            top_n_input = input("Tampilkan berapa HP teratas? [default 10]: ").strip()
            top_n = int(top_n_input) if top_n_input.isdigit() else 10

            result = recommend_smartphones(df, budget=budget, top_n=top_n)
            if result.empty:
                continue

            formatted_budget = format_currency(budget, dataset_scale=False)
            print(f"\n=== REKOMENDASI HP (budget <= {formatted_budget}) ===")
            print(
                tabulate(
                    format_result(result),
                    headers="keys",
                    tablefmt="fancy_grid",
                    showindex=False,
                )
            )
            print()


        elif choice == "3":
            print("Keluar dari aplikasi.")
            break
        else:
            print("Pilihan tidak dikenali.\n")


if __name__ == "__main__":
    main()
