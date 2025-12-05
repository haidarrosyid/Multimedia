import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import numpy as np

# ==============================
# KONFIGURASI
# ==============================

DATA_PATH = "Smartphones_cleaned_dataset.csv"
PRICE_FACTOR = 100  # angka di dataset × 100 = Rupiah

CRITERIA = [
    "price",
    "rating",
    "battery_capacity",
    "ram_capacity",
    "internal_memory",
    "primary_camera_rear",
    "primary_camera_front",
    "screen_size",
    "refresh_rate",
    "has_5g",
    "has_nfc",
    "fast_charging_available",
]

WEIGHTS = [
    4,   # price
    5,   # rating
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

# urutan kolom yang dipakai di tabel GUI
TABLE_COLUMNS = [
    "brand_name", "model", "price", "rating",
    "battery_capacity", "ram_capacity", "internal_memory",
    "primary_camera_rear", "primary_camera_front",
    "screen_size", "refresh_rate",
    "has_5g", "has_nfc", "fast_charging_available",
    "topsis_score",
]

# ==============================
# FUNGSI SPK (TOPSIS)
# ==============================

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    cols = ["brand_name", "model"] + CRITERIA
    df = df[cols].dropna().copy()
    df["price"] = df["price"].astype(float)
    return df


def topsis_scores(df: pd.DataFrame) -> pd.DataFrame:
    X = df[CRITERIA].astype(float).values

    norm = np.linalg.norm(X, axis=0)
    X_norm = X / norm

    w = np.array(WEIGHTS, dtype=float)
    w = w / w.sum()

    Xw = X_norm * w

    ideal_best = np.zeros(len(CRITERIA))
    ideal_worst = np.zeros(len(CRITERIA))

    for j, impact in enumerate(IMPACTS):
        if impact == '+':
            ideal_best[j] = Xw[:, j].max()
            ideal_worst[j] = Xw[:, j].min()
        else:
            ideal_best[j] = Xw[:, j].min()
            ideal_worst[j] = Xw[:, j].max()

    dist_best = np.linalg.norm(Xw - ideal_best, axis=1)
    dist_worst = np.linalg.norm(Xw - ideal_worst, axis=1)

    scores = dist_worst / (dist_best + dist_worst)

    df_out = df.copy()
    df_out["topsis_score"] = scores
    return df_out


def recommend_smartphones(df: pd.DataFrame, budget: int | None = None, top_n: int = 10) -> pd.DataFrame:
    df_scored = topsis_scores(df)

    if budget is not None:
        df_scored = df_scored[df_scored["price"] * PRICE_FACTOR <= budget].copy()

    if df_scored.empty:
        return df_scored

    df_scored = df_scored.sort_values("topsis_score", ascending=False)

    result = df_scored[TABLE_COLUMNS].head(top_n)
    return result


def format_result(df: pd.DataFrame) -> pd.DataFrame:
    """
    Format harga, boolean, dan skor supaya enak dibaca di GUI.
    Pastikan urutan kolom sesuai TABLE_COLUMNS.
    """
    # pastikan semua kolom ada
    temp = df.copy()
    for col in TABLE_COLUMNS:
        if col not in temp.columns:
            if col == "topsis_score":
                temp[col] = np.nan
            else:
                temp[col] = ""

    out = temp[TABLE_COLUMNS].copy()

    out["price"] = out["price"].apply(
        lambda x: "Rp " + f"{int(float(x) * PRICE_FACTOR):,}".replace(",", ".")
    )
    out["topsis_score"] = out["topsis_score"].astype(float).round(4)

    out["has_5g"] = out["has_5g"].apply(lambda x: "Ya" if bool(x) else "Tidak")
    out["has_nfc"] = out["has_nfc"].apply(lambda x: "Ya" if bool(x) else "Tidak")
    out["fast_charging_available"] = out["fast_charging_available"].apply(
        lambda x: "Ya" if bool(x) else "Tidak"
    )

    return out


# ==============================
# GUI TKINTER
# ==============================

class SPKApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("SPK Pemilihan HP Terbaik - TOPSIS")
        self.geometry("1150x500")

        try:
            self.df = load_data(DATA_PATH)
        except Exception as e:
            messagebox.showerror("Error", f"Gagal memuat data:\n{e}")
            self.destroy()
            return

        self.create_widgets()

    def create_widgets(self):
        frame_top = ttk.Frame(self, padding=10)
        frame_top.pack(side=tk.TOP, fill=tk.X)

        # Input budget
        ttk.Label(frame_top, text="Budget maksimum (Rp):").grid(row=0, column=0, sticky="w")
        self.entry_budget = ttk.Entry(frame_top, width=20)
        self.entry_budget.grid(row=0, column=1, padx=5)

        # Input top_n
        ttk.Label(frame_top, text="Jumlah HP ditampilkan:").grid(row=0, column=2, sticky="w")
        self.entry_topn = ttk.Entry(frame_top, width=5)
        self.entry_topn.insert(0, "10")
        self.entry_topn.grid(row=0, column=3, padx=5)

        # Tombol
        btn_all = ttk.Button(frame_top, text="10 HP Terbaik (tanpa budget)", command=self.show_all)
        btn_all.grid(row=0, column=4, padx=5)

        btn_budget = ttk.Button(frame_top, text="Rekomendasi Berdasarkan Budget", command=self.show_with_budget)
        btn_budget.grid(row=0, column=5, padx=5)

        btn_dataset = ttk.Button(frame_top, text="Lihat Semua Dataset", command=self.show_dataset)
        btn_dataset.grid(row=0, column=6, padx=5)

        # Tabel
        frame_table = ttk.Frame(self)
        frame_table.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.tree = ttk.Treeview(frame_table, columns=TABLE_COLUMNS, show="headings")

        headings = {
            "brand_name": "Brand",
            "model": "Model",
            "price": "Harga",
            "rating": "Rating",
            "battery_capacity": "Baterai (mAh)",
            "ram_capacity": "RAM (GB)",
            "internal_memory": "Memori (GB)",
            "primary_camera_rear": "Kamera Belakang (MP)",
            "primary_camera_front": "Kamera Depan (MP)",
            "screen_size": "Layar (inch)",
            "refresh_rate": "Refresh (Hz)",
            "has_5g": "5G",
            "has_nfc": "NFC",
            "fast_charging_available": "Fast Charge",
            "topsis_score": "Skor TOPSIS"
        }

        for col in TABLE_COLUMNS:
            self.tree.heading(col, text=headings[col])
            self.tree.column(col, anchor="center", width=90)

        self.tree.column("brand_name", width=90, anchor="w")
        self.tree.column("model", width=220, anchor="w")
        self.tree.column("price", width=110, anchor="e")

        vsb = ttk.Scrollbar(frame_table, orient="vertical", command=self.tree.yview)
        hsb = ttk.Scrollbar(frame_table, orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)

        self.tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")

        frame_table.rowconfigure(0, weight=1)
        frame_table.columnconfigure(0, weight=1)

    def clear_table(self):
        for item in self.tree.get_children():
            self.tree.delete(item)

    def fill_table(self, df: pd.DataFrame):
        self.clear_table()
        if df.empty:
            messagebox.showinfo("Info", "Tidak ada HP yang memenuhi kriteria.")
            return

        df_fmt = format_result(df)

        for _, row in df_fmt.iterrows():
            values = [row[col] for col in TABLE_COLUMNS]
            self.tree.insert("", tk.END, values=values)

    def get_topn(self) -> int:
        txt = self.entry_topn.get().strip()
        if not txt:
            return 10
        try:
            n = int(txt)
            return max(1, n)
        except ValueError:
            messagebox.showwarning("Peringatan", "Jumlah HP harus berupa angka. Dipakai nilai default 10.")
            return 10

    def show_all(self):
        top_n = self.get_topn()
        df_res = recommend_smartphones(self.df, budget=None, top_n=top_n)
        self.fill_table(df_res)

    def show_with_budget(self):
        budget_txt = self.entry_budget.get().strip()
        if not budget_txt:
            messagebox.showwarning("Peringatan", "Isi dulu budget maksimum.")
            return
        try:
            budget = int(budget_txt)
        except ValueError:
            messagebox.showerror("Error", "Budget harus berupa angka bulat (misal 5000000).")
            return

        top_n = self.get_topn()
        df_res = recommend_smartphones(self.df, budget=budget, top_n=top_n)
        self.fill_table(df_res)

    def show_dataset(self):
        """Menampilkan seluruh isi dataset (tanpa perhitungan TOPSIS, max 500 baris)."""
        df_all = self.df.copy()

        # kalau belum ada kolom topsis_score, tambahkan supaya format_result tidak error
        if "topsis_score" not in df_all.columns:
            df_all["topsis_score"] = np.nan

        df_all = df_all.head(500)
        self.fill_table(df_all)


if __name__ == "__main__":
    app = SPKApp()
    app.mainloop()
