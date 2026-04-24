from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data_cleaning import load_cleaned_workbook
from src.recommender import HybridGARecommender


def build_holdout(
    ratings: pd.DataFrame,
    behavior: pd.DataFrame,
    seed: int = 42,
    max_users: int = 180,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[int, set[int]], list[int]]:
    rng = np.random.default_rng(seed)
    ratings = ratings.copy().sort_values(["user_id", "product_id"]).reset_index(drop=True)
    behavior = behavior.copy().sort_values(["user_id", "product_id"]).reset_index(drop=True)

    behavior_pos = behavior[(behavior["clicked"] == 1) | (behavior["purchased"] == 1)][["user_id", "product_id"]]
    ratings_pos = ratings[ratings["rating"] >= 4][["user_id", "product_id"]]
    positives = pd.concat([behavior_pos, ratings_pos], ignore_index=True).drop_duplicates()
    relevant = positives.groupby("user_id")["product_id"].apply(set).to_dict()

    users = [uid for uid, items in relevant.items() if len(items) >= 3]
    users = sorted(users, key=lambda uid: len(relevant[uid]), reverse=True)
    if len(users) > max_users:
        users = sorted(rng.choice(np.array(users, dtype=int), size=max_users, replace=False).tolist())

    holdout: dict[int, set[int]] = {}
    for uid in users:
        items = sorted(relevant[uid])
        holdout_size = max(1, int(round(len(items) * 0.2)))
        selected = rng.choice(np.array(items, dtype=int), size=holdout_size, replace=False)
        holdout[int(uid)] = {int(x) for x in selected.tolist()}

    ratings_mask = ratings.apply(lambda row: int(row["product_id"]) not in holdout.get(int(row["user_id"]), set()), axis=1)
    behavior_mask = behavior.apply(lambda row: int(row["product_id"]) not in holdout.get(int(row["user_id"]), set()), axis=1)

    return (
        ratings.loc[ratings_mask].reset_index(drop=True),
        behavior.loc[behavior_mask].reset_index(drop=True),
        holdout,
        users,
    )


def evaluate_project(data_dir: str | Path = "data", output_path: str | Path | None = None) -> dict[str, Any]:
    data_dir = Path(data_dir)
    cleaned, _, _ = load_cleaned_workbook(data_dir)
    users = cleaned["users"]
    products = cleaned["products"]
    ratings = cleaned["ratings"]
    behavior = cleaned["behavior"]

    train_ratings, train_behavior, holdout, eval_users = build_holdout(ratings, behavior)
    recommender = HybridGARecommender(
        data_dir=data_dir,
        users_df=users,
        products_df=products,
        ratings_df=train_ratings,
        behavior_df=train_behavior,
        seed=42,
    )

    baseline_precision: list[float] = []
    optimized_precision: list[float] = []
    baseline_recall: list[float] = []
    optimized_recall: list[float] = []
    baseline_diversity: list[float] = []
    optimized_diversity: list[float] = []
    baseline_items, optimized_items = set(), set()
    baseline_hits = 0
    optimized_hits = 0
    examples = []

    for uid in eval_users:
        result = recommender.compare_recommendations(int(uid))
        truth = holdout[int(uid)]
        baseline = [int(item["product_id"]) for item in result["baseline"]]
        optimized = [int(item["product_id"]) for item in result["optimized"]]

        b_hits = len(set(baseline) & truth)
        o_hits = len(set(optimized) & truth)
        baseline_hits += b_hits
        optimized_hits += o_hits

        baseline_precision.append(b_hits / recommender.list_size)
        optimized_precision.append(o_hits / recommender.list_size)
        baseline_recall.append(b_hits / len(truth))
        optimized_recall.append(o_hits / len(truth))

        baseline_diversity.append(len({item["category"] for item in result["baseline"]}) / recommender.list_size)
        optimized_diversity.append(len({item["category"] for item in result["optimized"]}) / recommender.list_size)

        baseline_items.update(baseline)
        optimized_items.update(optimized)

        if len(examples) < 4:
            examples.append(
                {
                    "user_id": int(uid),
                    "truth": sorted(int(x) for x in truth),
                    "baseline": result["baseline"],
                    "optimized": result["optimized"],
                }
            )

    summary = {
        "evaluated_users": len(eval_users),
        "precision_at_5": {
            "baseline": round(float(np.mean(baseline_precision)), 4),
            "optimized": round(float(np.mean(optimized_precision)), 4),
        },
        "recall_at_5": {
            "baseline": round(float(np.mean(baseline_recall)), 4),
            "optimized": round(float(np.mean(optimized_recall)), 4),
        },
        "diversity_at_5": {
            "baseline": round(float(np.mean(baseline_diversity)), 4),
            "optimized": round(float(np.mean(optimized_diversity)), 4),
        },
        "coverage": {
            "baseline": round(len(baseline_items) / products["product_id"].nunique(), 4),
            "optimized": round(len(optimized_items) / products["product_id"].nunique(), 4),
        },
        "hit_rate": {
            "baseline": round(baseline_hits / len(eval_users), 4) if eval_users else 0.0,
            "optimized": round(optimized_hits / len(eval_users), 4) if eval_users else 0.0,
        },
        "examples": examples,
    }

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    return summary


if __name__ == "__main__":
    result = evaluate_project(output_path="docs/evaluation_summary.json")
    print(json.dumps(result, ensure_ascii=False, indent=2))
