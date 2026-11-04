from __future__ import annotations
from pathlib import Path
import pandas as pd


class DataCleaning:
    # Valid range for each numeric column: (bound, strict).
    # strict=True  -> value must be  > bound
    # strict=False -> value must be >= bound
    VALID_MIN = {
        "vehicle_age":   (0, False),
        "km_driven":     (0, False),
        "mileage":       (0, True),
        "engine":        (0, True),
        "max_power":     (0, True),
        "seats":         (0, True),
        "selling_price": (0, True),
    }

    def __init__(self, df: pd.DataFrame, target: str = "selling_price"):
        self.raw = df.copy()          # untouched copy, so you can always compare
        self.target = target
        self._df = df.copy()          # working copy that every step modifies
        self.data_cleaned: pd.DataFrame | None = None   # <- final result, in memory
        self.report: list[dict] = []  # audit trail of what each step did


    @classmethod
    def from_csv(cls, path: str | Path, target: str = "selling_price") -> "DataCleaning":
        return cls(pd.read_csv(path), target=target)

    def _log(self, step: str, before: int, after: int, note: str = "") -> None:
        self.report.append({"step": step, "rows_before": before,
                            "rows_after": after, "rows_removed": before - after,
                            "note": note})

    def report_df(self) -> pd.DataFrame:
        return pd.DataFrame(self.report)

    def _text_cols(self) -> list[str]:
        return [c for c in self._df.columns
                if pd.api.types.is_object_dtype(self._df[c])
                or pd.api.types.is_string_dtype(self._df[c])]

    def _numeric_cols(self) -> list[str]:
        return [c for c in self._df.columns if pd.api.types.is_numeric_dtype(self._df[c])]


    def drop_unnamed_columns(self) -> "DataCleaning":
        """Remove index columns that pandas/Excel leave behind ('Unnamed: 0')."""
        n_before = self._df.shape[1]
        self._df = self._df.loc[:, ~self._df.columns.astype(str).str.startswith("Unnamed")]
        self._log("drop_unnamed_columns", len(self._df), len(self._df),
                  f"{n_before - self._df.shape[1]} column(s) dropped")
        return self

    def standardize_column_names(self) -> "DataCleaning":
        """' Car Name ' -> 'car_name'."""
        self._df.columns = (self._df.columns.astype(str).str.strip().str.lower()
                            .str.replace(r"[\s\-]+", "_", regex=True))
        return self


    def standardize_text(self) -> "DataCleaning":
        """strip + lowercase + collapse repeated spaces.
        Fixes 'ISUZU' vs 'Isuzu'. NaN stays NaN."""
        for c in self._text_cols():
            s = self._df[c].astype("string")
            self._df[c] = (s.str.strip().str.lower()
                            .str.replace(r"\s+", " ", regex=True)
                            .astype(object))
        return self


    def fix_dtypes(self, numeric_cols: list[str] | None = None) -> "DataCleaning":
        """Force numeric columns to be numeric. Junk like 'N/A' or '12,000'
        becomes NaN (then handled by handle_missing_values)."""
        numeric_cols = numeric_cols or [c for c in self.VALID_MIN if c in self._df]
        for c in numeric_cols:
            if not pd.api.types.is_numeric_dtype(self._df[c]):
                self._df[c] = pd.to_numeric(
                    self._df[c].astype(str).str.replace(",", "", regex=False),
                    errors="coerce")
        return self


    def handle_missing_values(self, numeric_strategy: str = "median",
                              categorical_strategy: str = "mode") -> "DataCleaning":
        """
        - rows with a missing TARGET are dropped (never invent a label)
        - numeric features     -> median (or 'mean')
        - categorical features -> mode   (or 'unknown')
        Your raw.csv has 0 nulls, so today this changes nothing, but the class
        stays reusable for other data.
        """
        before = len(self._df)
        if self.target in self._df:
            self._df = self._df.dropna(subset=[self.target])

        filled = 0
        for c in self._numeric_cols():
            if c == self.target:
                continue
            n_null = int(self._df[c].isna().sum())
            if n_null:
                value = (self._df[c].median() if numeric_strategy == "median"
                         else self._df[c].mean())
                self._df[c] = self._df[c].fillna(value)
                filled += n_null
        for c in self._text_cols():
            n_null = int(self._df[c].isna().sum())
            if n_null:
                value = (self._df[c].mode().iloc[0] if categorical_strategy == "mode"
                         else "unknown")
                self._df[c] = self._df[c].fillna(value)
                filled += n_null

        self._log("handle_missing_values", before, len(self._df),
                  f"{filled} feature cell(s) imputed")
        return self



    def drop_duplicate_rows(self) -> "DataCleaning":
        """Drop rows identical in EVERY column, including the target.
        Two cars with identical specs but different prices are different
        listings, not duplicates."""
        before = len(self._df)
        self._df = self._df.drop_duplicates()
        self._log("drop_duplicate_rows", before, len(self._df))
        return self


    def drop_redundant_columns(self) -> "DataCleaning":
        """car_name is dropped only if it is exactly brand + ' ' + model."""
        if {"car_name", "brand", "model"} <= set(self._df.columns):
            mismatches = int((self._df["car_name"]
                              != self._df["brand"] + " " + self._df["model"]).sum())
            if mismatches == 0:
                self._df = self._df.drop(columns="car_name")
                self._log("drop_redundant_columns", len(self._df), len(self._df),
                          "car_name dropped (== brand + model)")
            else:
                self._log("drop_redundant_columns", len(self._df), len(self._df),
                          f"car_name KEPT: {mismatches} rows differ from brand+model")
        return self


    def remove_impossible_values(self) -> "DataCleaning":
        """Drop rows that break common sense (0 seats, negative km, price <= 0...)."""
        before = len(self._df)
        keep = pd.Series(True, index=self._df.index)
        for col, (bound, strict) in self.VALID_MIN.items():
            if col in self._df:
                keep &= (self._df[col] > bound) if strict else (self._df[col] >= bound)
        self._df = self._df[keep]
        self._log("remove_impossible_values", before, len(self._df))
        return self


    def handle_outliers(self, max_km_per_year: float = 150_000,
                        iqr_cols: list[str] | None = None, iqr_k: float = 3.0,
                        iqr_action: str = "cap") -> "DataCleaning":
        """
        Two kinds of outliers, handled differently on purpose:

        1) ERRORS (domain rule): km_driven / vehicle_age above `max_km_per_year`
           is not a real car (e.g. 3.8 million km on a 5-year-old XUV500).
           These rows are dropped.

        2) EXTREME-BUT-REAL values (a Ferrari's price, a 6.6 L engine) are NOT
           removed by default, because the model must learn that luxury cars
           exist. Pass iqr_cols=[...] to 'cap' or 'drop' values outside
           Q1 - k*IQR .. Q3 + k*IQR for the columns you choose.
           The target is never touched here; use a log transform in
           feature engineering instead.
        """
        before = len(self._df)
        if {"km_driven", "vehicle_age"} <= set(self._df.columns):
            km_per_year = self._df["km_driven"] / self._df["vehicle_age"].clip(lower=1)
            self._df = self._df[km_per_year <= max_km_per_year]
        self._log("outliers_domain_rule", before, len(self._df),
                  f"km/year > {max_km_per_year:,.0f}")

        for c in (iqr_cols or []):
            if c == self.target or c not in self._df:
                continue
            q1, q3 = self._df[c].quantile([0.25, 0.75])
            lo, hi = q1 - iqr_k * (q3 - q1), q3 + iqr_k * (q3 - q1)
            mask = (self._df[c] < lo) | (self._df[c] > hi)
            if iqr_action == "drop":
                b = len(self._df)
                self._df = self._df[~mask]
                self._log(f"outliers_iqr_drop[{c}]", b, len(self._df))
            else:
                self._df[c] = self._df[c].clip(lo, hi)
                self._log(f"outliers_iqr_cap[{c}]", len(self._df), len(self._df),
                          f"{int(mask.sum())} value(s) capped to [{lo:.1f}, {hi:.1f}]")
        return self

    def run(self, **outlier_kwargs) -> pd.DataFrame:
        """Run all steps in the right order and store the result in self.data_cleaned.

        Order matters:
          text first  -> 'ISUZU' and 'Isuzu' become equal BEFORE duplicates are found
          duplicates  -> before outliers, so repeated rows don't skew quantiles
          impossible  -> before outliers, so IQR isn't computed on garbage
        """
        self._log("start", len(self._df), len(self._df))
        (self.drop_unnamed_columns()
             .standardize_column_names()
             .standardize_text()
             .fix_dtypes()
             .handle_missing_values()
             .drop_duplicate_rows()
             .drop_redundant_columns()
             .remove_impossible_values()
             .handle_outliers(**outlier_kwargs))
        self.data_cleaned = self._df.reset_index(drop=True)
        return self.data_cleaned


if __name__ == "__main__":
    ROOT = Path(__file__).resolve().parents[1]
    cleaner = DataCleaning(pd.read_csv(ROOT/"data"/"raw.csv"))
    cleaned = cleaner.run()
    print(cleaner.report_df().to_string(index=False))
    print("\nFinal shape:", cleaned.shape)