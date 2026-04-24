from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


CLEANED_WORKBOOK_NAME = "cleaned_dataset.xlsx"
DATA_SHEETS = ("users", "products", "ratings", "behavior")
SUMMARY_SHEET = "summary"


def _normalize_headers(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame.copy()
    normalized.columns = [str(col).strip().lower() for col in normalized.columns]
    return normalized


def _require_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing required columns: {', '.join(missing)}")


def _coerce_binary(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce").fillna(0)
    return (numeric > 0).astype(int)


def find_behavior_file(data_dir: str | Path) -> Path:
    data_dir = Path(data_dir)
    direct = data_dir / "behavior_15500.xlsx"
    if direct.exists():
        return direct

    matches = sorted(data_dir.glob("behavior*.xlsx"))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"No behavior*.xlsx file found in {data_dir}")


def load_raw_datasets(data_dir: str | Path) -> tuple[dict[str, pd.DataFrame], Path]:
    data_dir = Path(data_dir)
    behavior_path = find_behavior_file(data_dir)
    frames = {
        "users": pd.read_excel(data_dir / "users.xlsx"),
        "products": pd.read_excel(data_dir / "products.xlsx"),
        "ratings": pd.read_excel(data_dir / "ratings.xlsx"),
        "behavior": pd.read_excel(behavior_path),
    }
    return frames, behavior_path


def raw_dataset_paths(data_dir: str | Path) -> list[Path]:
    data_dir = Path(data_dir)
    return [
        data_dir / "users.xlsx",
        data_dir / "products.xlsx",
        data_dir / "ratings.xlsx",
        find_behavior_file(data_dir),
    ]


def clean_users(users_raw: pd.DataFrame) -> pd.DataFrame:
    users = _normalize_headers(users_raw)
    if "country" not in users.columns and "location" in users.columns:
        users = users.rename(columns={"location": "country"})
    _require_columns(users, {"user_id", "age", "country"}, "users.xlsx")

    users = users[["user_id", "age", "country"]].copy()
    users["user_id"] = pd.to_numeric(users["user_id"], errors="coerce")
    users["age"] = pd.to_numeric(users["age"], errors="coerce")
    users["country"] = users["country"].fillna("").astype(str).str.strip()
    users.loc[users["country"].eq(""), "country"] = "Unknown"

    users = users.dropna(subset=["user_id"]).copy()
    users["user_id"] = users["user_id"].astype(int)

    valid_age = users["age"].dropna()
    median_age = int(valid_age.median()) if not valid_age.empty else 30
    users["age"] = users["age"].fillna(median_age).round().clip(lower=13, upper=100).astype(int)

    return users.drop_duplicates("user_id").sort_values("user_id").reset_index(drop=True)


def clean_products(products_raw: pd.DataFrame) -> pd.DataFrame:
    products = _normalize_headers(products_raw)
    _require_columns(products, {"product_id", "category", "price"}, "products.xlsx")

    products = products[["product_id", "category", "price"]].copy()
    products["product_id"] = pd.to_numeric(products["product_id"], errors="coerce")
    products["price"] = pd.to_numeric(products["price"], errors="coerce")
    products["category"] = (
        products["category"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
    )
    products.loc[products["category"].eq(""), "category"] = "Unknown"

    products = products.dropna(subset=["product_id", "price"]).copy()
    products = products[products["price"] >= 0].copy()
    products["product_id"] = products["product_id"].astype(int)
    products["price"] = products["price"].astype(float)

    return products.drop_duplicates("product_id").sort_values("product_id").reset_index(drop=True)


def clean_ratings(ratings_raw: pd.DataFrame, valid_users: set[int], valid_products: set[int]) -> pd.DataFrame:
    ratings = _normalize_headers(ratings_raw)
    _require_columns(ratings, {"user_id", "product_id", "rating"}, "ratings.xlsx")

    ratings = ratings[["user_id", "product_id", "rating"]].copy()
    ratings["user_id"] = pd.to_numeric(ratings["user_id"], errors="coerce")
    ratings["product_id"] = pd.to_numeric(ratings["product_id"], errors="coerce")
    ratings["rating"] = pd.to_numeric(ratings["rating"], errors="coerce")

    ratings = ratings.dropna(subset=["user_id", "product_id", "rating"]).copy()
    ratings["user_id"] = ratings["user_id"].astype(int)
    ratings["product_id"] = ratings["product_id"].astype(int)
    ratings = ratings[
        ratings["user_id"].isin(valid_users) & ratings["product_id"].isin(valid_products)
    ].copy()
    ratings["rating"] = ratings["rating"].clip(lower=1, upper=5)
    ratings = ratings.groupby(["user_id", "product_id"], as_index=False)["rating"].mean()
    ratings["rating"] = ratings["rating"].round(4)

    return ratings.sort_values(["user_id", "product_id"]).reset_index(drop=True)


def clean_behavior(
    behavior_raw: pd.DataFrame,
    valid_users: set[int],
    valid_products: set[int],
) -> pd.DataFrame:
    behavior = _normalize_headers(behavior_raw)
    _require_columns(behavior, {"user_id", "product_id", "viewed", "clicked", "purchased"}, "behavior.xlsx")

    behavior = behavior[["user_id", "product_id", "viewed", "clicked", "purchased"]].copy()
    behavior["user_id"] = pd.to_numeric(behavior["user_id"], errors="coerce")
    behavior["product_id"] = pd.to_numeric(behavior["product_id"], errors="coerce")
    behavior = behavior.dropna(subset=["user_id", "product_id"]).copy()
    behavior["user_id"] = behavior["user_id"].astype(int)
    behavior["product_id"] = behavior["product_id"].astype(int)
    behavior = behavior[
        behavior["user_id"].isin(valid_users) & behavior["product_id"].isin(valid_products)
    ].copy()

    for column in ["viewed", "clicked", "purchased"]:
        behavior[column] = _coerce_binary(behavior[column])

    behavior = behavior.groupby(["user_id", "product_id"], as_index=False)[
        ["viewed", "clicked", "purchased"]
    ].max()
    behavior["clicked"] = behavior[["clicked", "purchased"]].max(axis=1).astype(int)
    behavior["viewed"] = behavior[["viewed", "clicked"]].max(axis=1).astype(int)

    return behavior.sort_values(["user_id", "product_id"]).reset_index(drop=True)


def build_cleaned_datasets(data_dir: str | Path) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    raw_frames, behavior_path = load_raw_datasets(data_dir)

    users = clean_users(raw_frames["users"])
    products = clean_products(raw_frames["products"])
    valid_users = set(users["user_id"].tolist())
    valid_products = set(products["product_id"].tolist())
    ratings = clean_ratings(raw_frames["ratings"], valid_users, valid_products)
    behavior = clean_behavior(raw_frames["behavior"], valid_users, valid_products)

    cleaned = {
        "users": users,
        "products": products,
        "ratings": ratings,
        "behavior": behavior,
    }
    summary = {
        "source_behavior_file": behavior_path.name,
        "users_raw_rows": int(len(raw_frames["users"])),
        "users_after_cleaning": int(len(users)),
        "products_raw_rows": int(len(raw_frames["products"])),
        "products_after_cleaning": int(len(products)),
        "ratings_raw_rows": int(len(raw_frames["ratings"])),
        "ratings_after_cleaning": int(len(ratings)),
        "behavior_raw_rows": int(len(raw_frames["behavior"])),
        "behavior_after_cleaning": int(len(behavior)),
        "rows_removed_users": int(len(raw_frames["users"]) - len(users)),
        "rows_removed_products": int(len(raw_frames["products"]) - len(products)),
        "rows_removed_ratings": int(len(raw_frames["ratings"]) - len(ratings)),
        "rows_removed_behavior": int(len(raw_frames["behavior"]) - len(behavior)),
    }
    return cleaned, summary


def _summary_to_frame(summary: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(
        [{"metric": key, "value": value} for key, value in summary.items()],
        columns=["metric", "value"],
    )


def export_cleaned_workbook(
    cleaned: dict[str, pd.DataFrame],
    summary: dict[str, Any],
    output_path: str | Path,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        for sheet_name in DATA_SHEETS:
            cleaned[sheet_name].to_excel(writer, sheet_name=sheet_name, index=False)
        _summary_to_frame(summary).to_excel(writer, sheet_name=SUMMARY_SHEET, index=False)
    return output_path


def ensure_cleaned_workbook(
    data_dir: str | Path = "data",
    output_path: str | Path | None = None,
    force: bool = False,
) -> Path:
    data_dir = Path(data_dir)
    workbook_path = Path(output_path) if output_path is not None else data_dir / CLEANED_WORKBOOK_NAME

    raw_paths = raw_dataset_paths(data_dir)

    should_refresh = force or not workbook_path.exists()
    if not should_refresh:
        workbook_mtime = workbook_path.stat().st_mtime
        should_refresh = any(path.stat().st_mtime > workbook_mtime for path in raw_paths)

    if should_refresh:
        cleaned, summary = build_cleaned_datasets(data_dir)
        export_cleaned_workbook(cleaned, summary, workbook_path)

    return workbook_path


def _coerce_summary_value(value: Any) -> Any:
    if pd.isna(value):
        return None
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def load_cleaned_workbook(
    data_dir: str | Path = "data",
    workbook_path: str | Path | None = None,
    ensure_exists: bool = True,
    force_refresh: bool = False,
) -> tuple[dict[str, pd.DataFrame], dict[str, Any], Path]:
    data_dir = Path(data_dir)
    workbook_path = Path(workbook_path) if workbook_path is not None else data_dir / CLEANED_WORKBOOK_NAME
    if ensure_exists:
        workbook_path = ensure_cleaned_workbook(data_dir=data_dir, output_path=workbook_path, force=force_refresh)

    cleaned = {
        sheet_name: pd.read_excel(workbook_path, sheet_name=sheet_name)
        for sheet_name in DATA_SHEETS
    }
    summary_frame = pd.read_excel(workbook_path, sheet_name=SUMMARY_SHEET)
    summary = {
        str(row.metric): _coerce_summary_value(row.value)
        for row in summary_frame.itertuples(index=False)
    }
    return cleaned, summary, workbook_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean raw recommendation datasets and export a cached workbook.")
    parser.add_argument("--data-dir", default="data", help="Directory containing the raw Excel source files.")
    parser.add_argument("--output", default=None, help="Optional output workbook path.")
    parser.add_argument("--force", action="store_true", help="Regenerate the cleaned workbook even if it is up to date.")
    args = parser.parse_args()

    workbook_path = ensure_cleaned_workbook(data_dir=args.data_dir, output_path=args.output, force=args.force)
    _, summary, _ = load_cleaned_workbook(
        data_dir=args.data_dir,
        workbook_path=workbook_path,
        ensure_exists=False,
    )
    payload = {"cleaned_workbook": str(workbook_path), **summary}
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
