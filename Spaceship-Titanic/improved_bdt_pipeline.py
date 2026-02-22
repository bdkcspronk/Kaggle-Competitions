"""Dependency-free boosted decision tree pipeline for Spaceship Titanic.

This version can run in restricted environments without numpy/pandas/sklearn/xgboost.
It trains a lightweight gradient-boosted decision stump model and writes
`submission_improved.csv`.
"""

from __future__ import annotations

import csv
import math
import random
from dataclasses import dataclass
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parent
TRAIN_PATH = ROOT / "data" / "train.csv"
TEST_PATH = ROOT / "data" / "test.csv"
OUT_PATH = ROOT / "submission_improved.csv"

SPEND_COLS = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
CAT_COLS = ["HomePlanet", "Destination", "Deck", "Side", "Group_num", "Group_id", "Surname"]
NUM_COLS = [
    "Age", "VIP", "CryoSleep",
    "RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck",
    "Cabin_num", "TotalSpent", "ZeroSpend", "LogTotalSpent",
]
ALL_FEATURES = CAT_COLS + NUM_COLS
SEEDS = [42]


@dataclass
class Stump:
    feature: str
    threshold: float
    left_value: float
    right_value: float


class SimpleGBDT:
    def __init__(self, n_estimators: int = 25, learning_rate: float = 0.08, feature_subsample: float = 0.45, seed: int = 42):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.feature_subsample = feature_subsample
        self.seed = seed
        self.base_score = 0.0
        self.stumps: list[Stump] = []

    @staticmethod
    def _sigmoid(x: float) -> float:
        if x >= 0:
            z = math.exp(-x)
            return 1.0 / (1.0 + z)
        z = math.exp(x)
        return z / (1.0 + z)

    def fit(self, rows: list[dict[str, float]], y: list[int]) -> None:
        rng = random.Random(self.seed)
        p = min(max(sum(y) / max(len(y), 1), 1e-6), 1 - 1e-6)
        self.base_score = math.log(p / (1 - p))
        preds = [self.base_score for _ in rows]

        for _ in range(self.n_estimators):
            residuals = [yt - self._sigmoid(pr) for yt, pr in zip(y, preds)]
            features = ALL_FEATURES[:]
            rng.shuffle(features)
            take = max(3, int(len(features) * self.feature_subsample))
            cand_features = features[:take]
            stump = self._fit_best_stump(rows, residuals, cand_features)
            self.stumps.append(stump)
            for i, row in enumerate(rows):
                preds[i] += self.learning_rate * self._stump_predict(stump, row)

    def _fit_best_stump(self, rows: list[dict[str, float]], residuals: list[float], features: list[str]) -> Stump:
        best = None
        best_loss = float("inf")

        for feat in features:
            vals = [row[feat] for row in rows]
            uniq = sorted(set(vals))
            if len(uniq) <= 1:
                continue

            if len(uniq) > 12:
                thresholds = [uniq[int(i * (len(uniq) - 1) / 11)] for i in range(1, 11)]
            else:
                thresholds = [(uniq[i] + uniq[i + 1]) * 0.5 for i in range(len(uniq) - 1)]

            total_sum = sum(residuals)
            total_sq = sum(r * r for r in residuals)
            n = len(residuals)

            for thr in thresholds:
                left_n = 0
                left_sum = 0.0
                left_sq = 0.0
                for v, r in zip(vals, residuals):
                    if v <= thr:
                        left_n += 1
                        left_sum += r
                        left_sq += r * r
                right_n = n - left_n
                if left_n == 0 or right_n == 0:
                    continue

                right_sum = total_sum - left_sum
                right_sq = total_sq - left_sq
                left_mean = left_sum / left_n
                right_mean = right_sum / right_n
                left_loss = left_sq - 2 * left_mean * left_sum + left_n * left_mean * left_mean
                right_loss = right_sq - 2 * right_mean * right_sum + right_n * right_mean * right_mean
                loss = left_loss + right_loss

                if loss < best_loss:
                    best_loss = loss
                    best = Stump(feat, float(thr), float(left_mean), float(right_mean))

        if best is None:
            return Stump(ALL_FEATURES[0], 0.0, 0.0, 0.0)
        return best

    @staticmethod
    def _stump_predict(stump: Stump, row: dict[str, float]) -> float:
        return stump.left_value if row[stump.feature] <= stump.threshold else stump.right_value

    def predict_proba(self, rows: list[dict[str, float]]) -> list[float]:
        out = []
        for row in rows:
            score = self.base_score
            for stump in self.stumps:
                score += self.learning_rate * self._stump_predict(stump, row)
            out.append(self._sigmoid(score))
        return out


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def to_float(value: str, default: float = 0.0) -> float:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except ValueError:
        return default


def add_features(rows: list[dict[str, str]]) -> list[dict[str, str | float]]:
    out = []
    for r in rows:
        row = dict(r)
        pid = row.get("PassengerId", "")
        parts = pid.split("_") if "_" in pid else ["0", "0"]
        row["Group_num"], row["Group_id"] = parts[0], parts[1]

        cabin = row.get("Cabin") or "Unknown/0/Unknown"
        c = cabin.split("/")
        if len(c) != 3:
            c = ["Unknown", "0", "Unknown"]
        row["Deck"], row["Cabin_num"], row["Side"] = c[0] or "Unknown", c[1] or "0", c[2] or "Unknown"

        name = row.get("Name") or "Unknown"
        row["Surname"] = name.split()[-1] if name.strip() else "Unknown"

        total = 0.0
        for col in SPEND_COLS:
            v = to_float(row.get(col, ""), 0.0)
            row[col] = v
            total += v
        row["TotalSpent"] = total
        row["ZeroSpend"] = 1.0 if total == 0 else 0.0
        row["LogTotalSpent"] = math.log1p(total)

        row["Age"] = to_float(row.get("Age", ""), float("nan"))
        row["VIP"] = 1.0 if str(row.get("VIP", "False")).lower() == "true" else 0.0
        row["CryoSleep"] = 1.0 if str(row.get("CryoSleep", "False")).lower() == "true" else 0.0
        row["Cabin_num"] = to_float(row["Cabin_num"], float("nan"))

        row["HomePlanet"] = row.get("HomePlanet") or "Unknown"
        row["Destination"] = row.get("Destination") or "Unknown"
        row["Deck"] = row.get("Deck") or "Unknown"
        row["Side"] = row.get("Side") or "Unknown"

        out.append(row)
    return out


def fill_missing(train: list[dict], test: list[dict]) -> None:
    all_rows = train + test
    age_vals = [r["Age"] for r in all_rows if not math.isnan(r["Age"])]
    cabin_vals = [r["Cabin_num"] for r in all_rows if not math.isnan(r["Cabin_num"])]
    age_med = median(age_vals) if age_vals else 27.0
    cabin_med = median(cabin_vals) if cabin_vals else 300.0

    for r in all_rows:
        if math.isnan(r["Age"]):
            r["Age"] = age_med
        if math.isnan(r["Cabin_num"]):
            r["Cabin_num"] = cabin_med


def fit_categorical_maps(train_rows: list[dict]) -> dict[str, dict[str, int]]:
    maps = {}
    for c in CAT_COLS:
        vals = sorted({str(r[c]) for r in train_rows})
        maps[c] = {v: i for i, v in enumerate(vals)}
    return maps


def transform_rows(rows: list[dict], maps: dict[str, dict[str, int]], te: dict[str, float], te_default: float) -> list[dict[str, float]]:
    out = []
    for r in rows:
        x: dict[str, float] = {}
        for c in CAT_COLS:
            x[c] = float(maps[c].get(str(r[c]), -1))
        for c in NUM_COLS:
            x[c] = float(r[c])
        x["Surname_TE"] = te.get(str(r["Surname"]), te_default)
        out.append(x)
    return out


def fit_target_encoding(rows: list[dict], y: list[int], col: str) -> tuple[dict[str, float], float]:
    global_mean = sum(y) / max(len(y), 1)
    counts: dict[str, int] = {}
    sums: dict[str, float] = {}
    for r, t in zip(rows, y):
        k = str(r[col])
        counts[k] = counts.get(k, 0) + 1
        sums[k] = sums.get(k, 0.0) + t
    smoothing = 20.0
    mapping = {}
    for k in counts:
        cnt = counts[k]
        mean = sums[k] / cnt
        mapping[k] = (mean * cnt + global_mean * smoothing) / (cnt + smoothing)
    return mapping, global_mean


def group_kfold_indices(groups: list[str], n_splits: int = 5) -> list[tuple[list[int], list[int]]]:
    uniq = sorted(set(groups))
    buckets = [[] for _ in range(n_splits)]
    for i, g in enumerate(uniq):
        buckets[i % n_splits].append(g)

    folds = []
    for i in range(n_splits):
        val_groups = set(buckets[i])
        tr_idx, va_idx = [], []
        for idx, g in enumerate(groups):
            if g in val_groups:
                va_idx.append(idx)
            else:
                tr_idx.append(idx)
        folds.append((tr_idx, va_idx))
    return folds


def accuracy(y: list[int], p: list[float], thr: float) -> float:
    hit = 0
    for yt, pp in zip(y, p):
        if (pp >= thr) == bool(yt):
            hit += 1
    return hit / max(len(y), 1)


def find_best_threshold(y: list[int], p: list[float]) -> tuple[float, float]:
    best_t, best_a = 0.5, -1.0
    for i in range(121):
        t = 0.35 + i * (0.30 / 120.0)
        a = accuracy(y, p, t)
        if a > best_a:
            best_t, best_a = t, a
    return best_t, best_a


def main() -> None:
    train_raw = add_features(load_csv(TRAIN_PATH))
    test_raw = add_features(load_csv(TEST_PATH))
    fill_missing(train_raw, test_raw)

    y = [1 if str(r["Transported"]).lower() == "true" else 0 for r in train_raw]

    cat_maps = fit_categorical_maps(train_raw)
    te_map, te_default = fit_target_encoding(train_raw, y, "Surname")

    x_train = transform_rows(train_raw, cat_maps, te_map, te_default)
    x_test = transform_rows(test_raw, cat_maps, te_map, te_default)

    model = SimpleGBDT(seed=SEEDS[0])
    model.fit(x_train, y)

    train_pred = model.predict_proba(x_train)
    best_t, best_acc = find_best_threshold(y, train_pred)

    test_pred = model.predict_proba(x_test)

    with OUT_PATH.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["PassengerId", "Transported"])
        for r, p in zip(test_raw, test_pred):
            w.writerow([r["PassengerId"], "True" if p >= best_t else "False"])

    print(f"Train accuracy (threshold-tuned): {best_acc:.5f}")
    print(f"Best threshold: {best_t:.4f}")
    print(f"Saved: {OUT_PATH}")


if __name__ == "__main__":
    main()
