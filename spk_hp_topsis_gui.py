import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import numpy as np

DATA_PATH = "Smartphones_cleaned_dataset.csv"
PRICE_FACTOR = 100

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

WEIGHTS = [4, 5, 4, 3, 2, 3, 2, 1.5, 1.5, 2, 1, 1.5]

IMPACTS = ['-', '+', '+', '+', '+', '+', '+', '+', '+', '+', '+', '+']

TABLE_COLUMNS = [
    "brand_name", "model", "price", "rating",
    "battery_capacity", "ram_capacity", "internal_memory",
    "primary_camera_rear", "primary_camera_front",
    "screen_size", "refresh_rate",
    "has_5g", "has_nfc", "fast_charging_available",
    "topsis_score",
]


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


def recommend_smartphones(
    df: pd.DataFrame,
    budget: int | None = None,
    top_n: int = 10,
    min_rating: float | None = None
) -> pd.DataFrame:
    df_scored = topsis_scores(df)

    if budget is not None:
        df_scored = df_scored[df_scored["price"] * PRICE_FACTOR <= budget].copy()

    if min_rating is not None:
        df_scored = df_scored[df_scored["rating"] >= min_rating].copy()

    if df_scored.empty:
        return df_scored

    df_scored = df_scored.sort_values("topsis_score", ascending=False)
    return df_scored[TABLE_COLUMNS].head(top_n)


def format_currency(value: float) -> str:
    try:
        amount = int(float(value) * PRICE_FACTOR)
    except (TypeError, ValueError):
        return "-"
    return "Rp " + f"{amount:,}".replace(",", ".")


def format_result(df: pd.DataFrame) -> pd.DataFrame:
    temp = df.copy()
    for col in TABLE_COLUMNS:
        if col not in temp.columns:
            temp[col] = np.nan if col == "topsis_score" else ""

    out = temp[TABLE_COLUMNS].copy()

    def format_decimal(val, decimals=2, suffix=""):
        try:
            num = float(val)
        except (TypeError, ValueError):
            return "-"
        return f"{num:.{decimals}f}{suffix}"

    def format_integer(val, suffix=""):
        try:
            num = int(float(val))
        except (TypeError, ValueError):
            return "-"
        return f"{num}{suffix}"

    def bool_label(val: object) -> str:
        if pd.isna(val):
            return "Tidak"
        try:
            return "Ya" if float(val) > 0 else "Tidak"
        except (TypeError, ValueError):
            return "Tidak"

    def score_label(val: object) -> str:
        if pd.isna(val):
            return "-"
        try:
            return f"{float(val):.4f}"
        except (TypeError, ValueError):
            return "-"

    out["price"] = out["price"].apply(format_currency)
    out["rating"] = out["rating"].apply(lambda v: format_decimal(v, 2))
    out["battery_capacity"] = out["battery_capacity"].apply(lambda v: format_integer(v, " mAh"))
    out["ram_capacity"] = out["ram_capacity"].apply(lambda v: format_integer(v, " GB"))
    out["internal_memory"] = out["internal_memory"].apply(lambda v: format_integer(v, " GB"))
    out["primary_camera_rear"] = out["primary_camera_rear"].apply(lambda v: format_integer(v, " MP"))
    out["primary_camera_front"] = out["primary_camera_front"].apply(lambda v: format_integer(v, " MP"))
    out["screen_size"] = out["screen_size"].apply(lambda v: format_decimal(v, 2, "\""))
    out["refresh_rate"] = out["refresh_rate"].apply(lambda v: format_integer(v, " Hz"))
    out["has_5g"] = out["has_5g"].apply(bool_label)
    out["has_nfc"] = out["has_nfc"].apply(bool_label)
    out["fast_charging_available"] = out["fast_charging_available"].apply(bool_label)
    out["topsis_score"] = out["topsis_score"].apply(score_label)

    return out


class SPKApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("SPK Pemilihan HP Terbaik - TOPSIS")
        self.geometry("1220x520")

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

        ttk.Label(frame_top, text="Budget maksimum (Rp):").grid(row=0, column=0, sticky="w")
        self.entry_budget = ttk.Entry(frame_top, width=18)
        self.entry_budget.grid(row=0, column=1, padx=5)

        ttk.Label(frame_top, text="Jumlah HP ditampilkan:").grid(row=0, column=2, sticky="w")
        self.entry_topn = ttk.Entry(frame_top, width=6)
        self.entry_topn.insert(0, "10")
        self.entry_topn.grid(row=0, column=3, padx=5)

        ttk.Label(frame_top, text="Rating minimum:").grid(row=0, column=4, sticky="w")
        self.entry_rating_min = ttk.Entry(frame_top, width=6)
        self.entry_rating_min.insert(0, "4.0")
        self.entry_rating_min.grid(row=0, column=5, padx=5)

        btn_all = ttk.Button(frame_top, text="10 HP Terbaik (tanpa budget)", command=self.show_all)
        btn_all.grid(row=0, column=6, padx=5)

        btn_recommend = ttk.Button(frame_top, text="Rekomendasikan", command=self.show_recommendation)
        btn_recommend.grid(row=0, column=7, padx=5)

        btn_dataset = ttk.Button(frame_top, text="Lihat Semua Dataset", command=self.show_dataset)
        btn_dataset.grid(row=0, column=8, padx=5)

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
            self.tree.column(col, anchor="center", width=95)

        self.tree.column("brand_name", width=110, anchor="w")
        self.tree.column("model", width=220, anchor="w")
        self.tree.column("price", width=125, anchor="e")

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
        column_order = self.tree["columns"]

        for _, row in df_fmt.iterrows():
            values = [row[col] for col in column_order]
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

    def get_min_rating(self) -> float | None:
        txt = self.entry_rating_min.get().strip()
        if not txt:
            return None
        try:
            value = float(txt)
            return max(0.0, value)
        except ValueError:
            messagebox.showwarning("Peringatan", "Rating minimum tidak valid. Filter rating diabaikan.")
            return None

    def show_all(self):
        top_n = self.get_topn()
        min_rating = self.get_min_rating()
        df_res = recommend_smartphones(self.df, budget=None, top_n=top_n, min_rating=min_rating)
        self.fill_table(df_res)

    def show_recommendation(self):
        budget_txt = self.entry_budget.get().strip()
        budget: int | None = None
        if budget_txt:
            try:
                budget = int(budget_txt)
            except ValueError:
                messagebox.showerror("Error", "Budget harus berupa angka bulat (misal 5000000).")
                return

        top_n = self.get_topn()
        min_rating = self.get_min_rating()
        df_res = recommend_smartphones(self.df, budget=budget, top_n=top_n, min_rating=min_rating)
        self.fill_table(df_res)

    def show_dataset(self):
        df_all = self.df.copy()
        df_all["topsis_score"] = np.nan
        self.fill_table(df_all.head(500))


if __name__ == "__main__":
    app = SPKApp()
    app.mainloop()