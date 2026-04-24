from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data_cleaning import load_cleaned_workbook


class HybridGARecommender:
    def __init__(
        self,
        data_dir: str | Path = "data",
        users_df: pd.DataFrame | None = None,
        products_df: pd.DataFrame | None = None,
        ratings_df: pd.DataFrame | None = None,
        behavior_df: pd.DataFrame | None = None,
        list_size: int = 5,
        seed: int = 42,
        ga_generations: int = 18,
        population_size: int = 28,
        mutation_rate: float = 0.28,
        neighbor_count: int = 25,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.list_size = list_size
        self.seed = seed
        self.ga_generations = ga_generations
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.neighbor_count = neighbor_count
        self.rng = np.random.default_rng(seed)
        self.cleaning_summary: dict[str, Any] = {}
        self.cleaned_workbook_path: Path | None = None

        if users_df is None:
            cleaned, summary, workbook_path = load_cleaned_workbook(self.data_dir)
            self.users_raw = cleaned["users"]
            self.products_raw = cleaned["products"]
            self.ratings_raw = cleaned["ratings"]
            self.behavior_raw = cleaned["behavior"]
            self.cleaning_summary = summary
            self.cleaned_workbook_path = workbook_path
        else:
            self.users_raw = users_df.copy()
            self.products_raw = products_df.copy()
            self.ratings_raw = ratings_df.copy()
            self.behavior_raw = behavior_df.copy()

        self._prepare()

    def _prepare(self) -> None:
        self.users = self.users_raw.copy().sort_values("user_id").reset_index(drop=True)
        self.users["user_id"] = self.users["user_id"].astype(int)
        self.users["age"] = self.users["age"].astype(int)
        self.users["country"] = self.users["country"].astype(str)

        self.products = self.products_raw.copy().sort_values("product_id").reset_index(drop=True)
        self.products["product_id"] = self.products["product_id"].astype(int)
        self.products["price"] = self.products["price"].astype(float)
        self.products["category"] = self.products["category"].astype(str)
        self.products["name"] = self.products.apply(
            lambda row: f"{row['category']} Item {int(row['product_id'])}",
            axis=1,
        )

        self.ratings = self.ratings_raw.copy().sort_values(["user_id", "product_id"]).reset_index(drop=True)
        self.ratings["user_id"] = self.ratings["user_id"].astype(int)
        self.ratings["product_id"] = self.ratings["product_id"].astype(int)
        self.ratings["rating"] = self.ratings["rating"].astype(float)

        self.behavior = self.behavior_raw.copy().sort_values(["user_id", "product_id"]).reset_index(drop=True)
        self.behavior["user_id"] = self.behavior["user_id"].astype(int)
        self.behavior["product_id"] = self.behavior["product_id"].astype(int)
        for column in ["viewed", "clicked", "purchased"]:
            self.behavior[column] = self.behavior[column].astype(int)

        self.categories = sorted(self.products["category"].unique().tolist())
        self.user_ids = self.users["user_id"].astype(int).tolist()
        self.product_ids = self.products["product_id"].astype(int).tolist()
        self.user_to_idx = {uid: idx for idx, uid in enumerate(self.user_ids)}
        self.product_to_idx = {pid: idx for idx, pid in enumerate(self.product_ids)}
        self.category_to_idx = {cat: idx for idx, cat in enumerate(self.categories)}

        self.product_meta = self.products.set_index("product_id")[["name", "category", "price"]]
        self.global_price_mean = float(self.products["price"].mean())

        self.behavior_enriched = self.behavior.merge(self.products, on="product_id", how="left")
        self.ratings_enriched = self.ratings.merge(self.products, on="product_id", how="left")

        self._build_product_features()
        self._build_user_profiles()
        self._build_neighbors()

    def _build_product_features(self) -> None:
        prod = self.products.copy()
        behavior = self.behavior.copy()
        behavior["signal"] = behavior["viewed"] * 0.15 + behavior["clicked"] * 0.35 + behavior["purchased"] * 0.50
        behavior["positive"] = behavior["clicked"] * 0.45 + behavior["purchased"] * 0.55

        product_behavior = behavior.groupby("product_id", as_index=False).agg(
            behavior_signal=("signal", "sum"),
            positive_signal=("positive", "sum"),
        )
        product_rating = self.ratings.groupby("product_id", as_index=False)["rating"].mean()
        product_rating.rename(columns={"rating": "avg_rating"}, inplace=True)

        prod = prod.merge(product_behavior, on="product_id", how="left").merge(product_rating, on="product_id", how="left")
        prod.fillna({"behavior_signal": 0.0, "positive_signal": 0.0, "avg_rating": 3.0}, inplace=True)

        self.product_frame = prod.set_index("product_id")
        self.product_popularity = self._normalize(prod["positive_signal"].to_numpy(dtype=float))
        self.product_rating_quality = ((prod["avg_rating"].to_numpy(dtype=float) - 1.0) / 4.0).clip(0, 1)
        self.product_categories = prod["category"].to_numpy(dtype=str)
        self.product_prices = prod["price"].to_numpy(dtype=float)

    def _build_user_profiles(self) -> None:
        n_users = len(self.user_ids)
        n_products = len(self.product_ids)
        n_categories = len(self.categories)

        self.user_item_scores = np.zeros((n_users, n_products), dtype=np.float32)
        category_pref = np.zeros((n_users, n_categories), dtype=float)
        price_sum = np.zeros(n_users, dtype=float)
        price_weight = np.zeros(n_users, dtype=float)

        self.user_purchased: dict[int, set[int]] = {uid: set() for uid in self.user_ids}
        self.user_interacted: dict[int, set[int]] = {uid: set() for uid in self.user_ids}
        self.user_ignored: dict[int, set[int]] = {uid: set() for uid in self.user_ids}
        self.user_low_rated: dict[int, set[int]] = {uid: set() for uid in self.user_ids}
        self.user_negative_categories: dict[int, set[str]] = {uid: set() for uid in self.user_ids}

        for row in self.behavior_enriched.itertuples(index=False):
            uidx = self.user_to_idx[int(row.user_id)]
            pidx = self.product_to_idx[int(row.product_id)]
            cidx = self.category_to_idx[str(row.category)]

            signal = 0.15 * int(row.viewed) + 0.35 * int(row.clicked) + 0.50 * int(row.purchased)
            category_pref[uidx, cidx] += signal
            self.user_item_scores[uidx, pidx] += (
                0.15 * int(row.viewed) + 0.40 * int(row.clicked) + 0.55 * int(row.purchased)
            )
            price_sum[uidx] += float(row.price) * (
                0.2 * int(row.viewed) + 0.4 * int(row.clicked) + 0.9 * int(row.purchased)
            )
            price_weight[uidx] += 0.2 * int(row.viewed) + 0.4 * int(row.clicked) + 0.9 * int(row.purchased)

            self.user_interacted[int(row.user_id)].add(int(row.product_id))
            if int(row.viewed) == 1 and int(row.clicked) == 0 and int(row.purchased) == 0:
                self.user_ignored[int(row.user_id)].add(int(row.product_id))
            if int(row.purchased) == 1:
                self.user_purchased[int(row.user_id)].add(int(row.product_id))

        for row in self.ratings_enriched.itertuples(index=False):
            uidx = self.user_to_idx[int(row.user_id)]
            pidx = self.product_to_idx[int(row.product_id)]
            cidx = self.category_to_idx[str(row.category)]

            rating_signal = (float(row.rating) - 3.0) / 2.0
            category_pref[uidx, cidx] += 0.75 * rating_signal
            self.user_item_scores[uidx, pidx] += 0.45 * max(rating_signal, -0.6)

            if float(row.rating) >= 3.0:
                price_sum[uidx] += float(row.price) * (float(row.rating) / 5.0)
                price_weight[uidx] += float(row.rating) / 5.0

            self.user_interacted[int(row.user_id)].add(int(row.product_id))
            if float(row.rating) <= 2.0:
                self.user_low_rated[int(row.user_id)].add(int(row.product_id))
                self.user_negative_categories[int(row.user_id)].add(str(row.category))

        category_pref = np.where(category_pref < 0, category_pref * 0.4, category_pref)
        shifted = category_pref - category_pref.min(axis=1, keepdims=True) + 1e-6
        self.category_pref = shifted / shifted.sum(axis=1, keepdims=True)

        self.user_preferred_price = np.divide(price_sum, np.where(price_weight == 0, 1.0, price_weight))
        self.user_preferred_price = np.where(price_weight == 0, self.global_price_mean, self.user_preferred_price)
        self.user_price_range = np.maximum(160.0, self.user_preferred_price * 0.30)

        profile = self.users.copy()
        profile["preferred_price"] = self.user_preferred_price
        self.user_profile_frame = profile.set_index("user_id")

    def _build_neighbors(self) -> None:
        age_scaled = self._normalize(self.users["age"].to_numpy(dtype=float))
        country_dummies = (
            pd.get_dummies(self.users["country"])
            .reindex(self.user_ids, fill_value=0)
            .to_numpy(dtype=float)
        )
        price_scaled = self._normalize(self.user_preferred_price)

        profile_matrix = np.concatenate(
            [self.category_pref, age_scaled.reshape(-1, 1), price_scaled.reshape(-1, 1), country_dummies],
            axis=1,
        )
        norms = np.linalg.norm(profile_matrix, axis=1, keepdims=True)
        profile_matrix = np.divide(profile_matrix, np.where(norms == 0, 1.0, norms))

        similarity = profile_matrix @ profile_matrix.T
        np.fill_diagonal(similarity, 0.0)
        self.user_similarity = similarity
        self.neighbor_product_scores = np.zeros_like(self.user_item_scores, dtype=float)
        fallback = self._normalize(self.user_item_scores.sum(axis=0))

        for idx in range(len(self.user_ids)):
            row = similarity[idx]
            neigh = np.argpartition(row, -self.neighbor_count)[-self.neighbor_count :]
            weights = np.clip(row[neigh], 0.0, None)
            if weights.sum() == 0:
                self.neighbor_product_scores[idx] = fallback
            else:
                self.neighbor_product_scores[idx] = (
                    (weights[:, None] * self.user_item_scores[neigh]).sum(axis=0) / weights.sum()
                )

        self.neighbor_product_scores = self._normalize_rows(self.neighbor_product_scores)

    def dataset_summary(self) -> dict[str, Any]:
        users_before = int(self.cleaning_summary.get("users_raw_rows", len(self.users_raw)))
        products_before = int(self.cleaning_summary.get("products_raw_rows", len(self.products_raw)))
        ratings_before = int(self.cleaning_summary.get("ratings_raw_rows", len(self.ratings_raw)))
        behavior_before = int(self.cleaning_summary.get("behavior_raw_rows", len(self.behavior_raw)))
        users_after = int(self.cleaning_summary.get("users_after_cleaning", len(self.users)))
        products_after = int(self.cleaning_summary.get("products_after_cleaning", len(self.products)))
        ratings_after = int(self.cleaning_summary.get("ratings_after_cleaning", len(self.ratings)))
        behavior_after = int(self.cleaning_summary.get("behavior_after_cleaning", len(self.behavior)))

        return {
            "users": users_after,
            "users_before_cleaning": users_before,
            "products": products_after,
            "products_before_cleaning": products_before,
            "ratings": ratings_before,
            "ratings_after_cleaning": ratings_after,
            "behavior_rows": behavior_before,
            "behavior_after_cleaning": behavior_after,
            "categories": self.categories,
            "countries": sorted(self.users["country"].unique().tolist()),
            "age_range": [int(self.users["age"].min()), int(self.users["age"].max())],
            "price_range": [
                float(self.products["price"].min()),
                float(self.products["price"].max()),
            ],
            "cleaned_workbook": self.cleaned_workbook_path.name if self.cleaned_workbook_path is not None else None,
            "duplicates_removed": {
                "users": users_before - users_after,
                "products": products_before - products_after,
                "ratings": ratings_before - ratings_after,
                "behavior": behavior_before - behavior_after,
            },
        }

    def get_user_profile_summary(self, user_id: int) -> dict[str, Any]:
        uidx = self.user_to_idx[user_id]
        scores = self.category_pref[uidx]
        order = np.argsort(scores)[::-1][:3]
        row = self.user_profile_frame.loc[user_id]

        return {
            "user_id": int(user_id),
            "age": int(row["age"]),
            "country": str(row["country"]),
            "preferred_price": round(float(self.user_preferred_price[uidx]), 2),
            "price_range": round(float(self.user_price_range[uidx]), 2),
            "top_categories": [
                {"category": self.categories[idx], "weight": round(float(scores[idx]), 4)}
                for idx in order
            ],
            "purchased_count": len(self.user_purchased[user_id]),
            "interactions_count": len(self.user_interacted[user_id]),
        }

    def compare_recommendations(
        self,
        user_id: int,
        ga_overrides: dict[str, int | float] | None = None,
    ) -> dict[str, Any]:
        ga_overrides = ga_overrides or {}
        candidate_ids, final_scores, parts = self._baseline_rank(user_id)
        baseline = candidate_ids[: self.list_size]
        ga_result = self._run_ga(
            user_id,
            candidate_ids,
            final_scores,
            parts,
            generations=int(ga_overrides.get("generations", self.ga_generations)),
            population_size=int(ga_overrides.get("population_size", self.population_size)),
            mutation_rate=float(ga_overrides.get("mutation_rate", self.mutation_rate)),
        )
        return {
            "user_profile": self.get_user_profile_summary(user_id),
            "baseline": self._format_items(user_id, baseline, parts, label="Baseline"),
            "optimized": self._format_items(user_id, ga_result["product_ids"], parts, label="Optimized"),
            "candidate_pool_size": len(candidate_ids),
            "best_fitness": ga_result["best_fitness"],
            "ga_trace": ga_result["history"],
            "ga_settings": ga_result["settings"],
        }

    def recommend(
        self,
        user_id: int,
        use_ga: bool = True,
        ga_overrides: dict[str, int | float] | None = None,
    ) -> list[dict[str, Any]]:
        result = self.compare_recommendations(user_id, ga_overrides=ga_overrides)
        return result["optimized" if use_ga else "baseline"]

    def sample_user_ids(self, count: int = 8) -> list[int]:
        order = (
            self.behavior.groupby("user_id")["product_id"]
            .count()
            .sort_values(ascending=False)
            .index.tolist()
        )
        return [int(uid) for uid in order[:count]]

    def search_products(self, query: str, limit: int = 24) -> list[dict[str, Any]]:
        query = query.strip().lower()
        if not query:
            return []
        matches = self.product_frame.reset_index()
        mask = (
            matches["name"].str.lower().str.contains(query, na=False)
            | matches["category"].str.lower().str.contains(query, na=False)
            | matches["product_id"].astype(str).str.contains(query, na=False)
        )
        subset = matches.loc[mask].head(limit)
        return self._records_from_frame(subset)

    def random_products(self, count: int = 8) -> list[dict[str, Any]]:
        ranked = self.product_frame.reset_index().copy()
        ranked["guest_score"] = 0.65 * ranked["avg_rating"] + 0.35 * ranked["positive_signal"]
        sample = ranked.sort_values("guest_score", ascending=False).head(max(count * 3, count))
        chosen = sample.sample(n=min(count, len(sample)), random_state=self.seed)
        return self._records_from_frame(chosen)

    def export_summary(self, output_path: str | Path) -> None:
        payload = {
            "dataset": self.dataset_summary(),
            "paper": {
                "title": "Hybrid and optimized product recommendation workflow for SmartShop",
                "method": "Content, collaborative, price alignment, popularity, and genetic optimization",
            },
        }
        Path(output_path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _baseline_rank(
        self,
        user_id: int,
        pool_size: int = 28,
    ) -> tuple[list[int], np.ndarray, dict[str, np.ndarray]]:
        uidx = self.user_to_idx[user_id]
        cat_score = self.category_pref[uidx]
        pref_price = self.user_preferred_price[uidx]
        price_range = self.user_price_range[uidx]

        content = np.array(
            [cat_score[self.category_to_idx[cat]] for cat in self.product_categories],
            dtype=float,
        )
        price = np.exp(-np.abs(self.product_prices - pref_price) / max(price_range, 1.0))
        collaborative = self.neighbor_product_scores[uidx]
        popularity = self.product_popularity
        rating = self.product_rating_quality
        final = 0.42 * content + 0.18 * price + 0.20 * collaborative + 0.10 * popularity + 0.10 * rating

        penalties = np.zeros_like(final)
        negative_categories = self.user_negative_categories[user_id]
        for pid in self.user_purchased[user_id]:
            penalties[self.product_to_idx[pid]] += 2.0
        for pid in self.user_low_rated[user_id]:
            penalties[self.product_to_idx[pid]] += 0.50
        for pid in self.user_ignored[user_id]:
            penalties[self.product_to_idx[pid]] += 0.18
        for idx, cat in enumerate(self.product_categories):
            if cat in negative_categories:
                penalties[idx] += 0.12

        final = np.where(final - penalties < 0, 0, final - penalties)
        ranked = np.argsort(final)[::-1]

        candidate_ids = []
        for idx in ranked:
            pid = self.product_ids[idx]
            if pid in self.user_purchased[user_id]:
                continue
            candidate_ids.append(pid)
            if len(candidate_ids) == pool_size:
                break

        return candidate_ids, final, {
            "content": content,
            "price": price,
            "collaborative": collaborative,
            "popularity": popularity,
            "rating": rating,
            "final": final,
        }

    def _run_ga(
        self,
        user_id: int,
        candidate_ids: list[int],
        final_scores: np.ndarray,
        parts: dict[str, np.ndarray],
        generations: int | None = None,
        population_size: int | None = None,
        mutation_rate: float | None = None,
    ) -> dict[str, Any]:
        generations = max(1, int(self.ga_generations if generations is None else generations))
        population_size = max(6, int(self.population_size if population_size is None else population_size))
        mutation_rate = float(self.mutation_rate if mutation_rate is None else mutation_rate)
        mutation_rate = min(max(mutation_rate, 0.0), 1.0)

        if len(candidate_ids) <= self.list_size:
            best_fitness = float(
                np.mean([final_scores[self.product_to_idx[pid]] for pid in candidate_ids])
            ) if candidate_ids else 0.0
            return {
                "product_ids": candidate_ids,
                "best_fitness": round(best_fitness, 4),
                "history": [],
                "settings": {
                    "generations": generations,
                    "population_size": population_size,
                    "mutation_rate": round(mutation_rate, 3),
                },
            }

        weights = np.array([final_scores[self.product_to_idx[pid]] for pid in candidate_ids], dtype=float)
        if weights.sum() == 0:
            weights = np.ones_like(weights)
        weights = weights / weights.sum()
        weight_by_id = {pid: float(weight) for pid, weight in zip(candidate_ids, weights)}

        population = [candidate_ids[: self.list_size]]
        while len(population) < population_size:
            sample = self.rng.choice(candidate_ids, size=self.list_size, replace=False, p=weights)
            population.append([int(x) for x in sample.tolist()])

        best = population[0]
        best_score = self._fitness(user_id, best, final_scores, parts)
        elite = max(2, population_size // 5)
        history: list[dict[str, float]] = []

        for generation in range(generations):
            scored = [(chrom, self._fitness(user_id, chrom, final_scores, parts)) for chrom in population]
            scored.sort(key=lambda item: item[1], reverse=True)
            history.append(
                {
                    "generation": float(generation),
                    "best_fitness": round(float(scored[0][1]), 4),
                    "mean_fitness": round(float(np.mean([score for _, score in scored])), 4),
                }
            )
            if scored[0][1] > best_score:
                best = list(scored[0][0])
                best_score = float(scored[0][1])

            next_population = [list(chrom) for chrom, _ in scored[:elite]]
            while len(next_population) < population_size:
                parent_a = self._pick_parent(scored)
                parent_b = self._pick_parent(scored)
                child = self._crossover(parent_a, parent_b, candidate_ids)
                child = self._mutate(child, candidate_ids, weight_by_id, mutation_rate)
                next_population.append(child)
            population = next_population

        final_scored = [(chrom, self._fitness(user_id, chrom, final_scores, parts)) for chrom in population]
        final_scored.sort(key=lambda item: item[1], reverse=True)
        final_mean = float(np.mean([score for _, score in final_scored]))
        history.append(
            {
                "generation": float(generations),
                "best_fitness": round(float(final_scored[0][1]), 4),
                "mean_fitness": round(final_mean, 4),
            }
        )
        if final_scored[0][1] > best_score:
            best = list(final_scored[0][0])
            best_score = float(final_scored[0][1])

        return {
            "product_ids": best,
            "best_fitness": round(best_score, 4),
            "history": history,
            "settings": {
                "generations": generations,
                "population_size": population_size,
                "mutation_rate": round(mutation_rate, 3),
            },
        }

    def _fitness(
        self,
        user_id: int,
        chromosome: list[int],
        final_scores: np.ndarray,
        parts: dict[str, np.ndarray],
    ) -> float:
        unique = list(dict.fromkeys(chromosome))
        if len(unique) < self.list_size:
            return -1.0

        idxs = np.array([self.product_to_idx[pid] for pid in unique], dtype=int)
        cats = [self.product_categories[idx] for idx in idxs]
        prices = self.product_prices[idxs]

        relevance = float(np.mean(final_scores[idxs]))
        content = float(np.mean(parts["content"][idxs]))
        collaborative = float(np.mean(parts["collaborative"][idxs]))
        diversity = len(set(cats)) / self.list_size
        novelty = float(np.mean(1.0 - parts["popularity"][idxs]))

        pref_price = self.user_preferred_price[self.user_to_idx[user_id]]
        price_alignment = 1.0 - min(
            abs(float(prices.mean()) - pref_price) / max(self.user_price_range[self.user_to_idx[user_id]], 1.0),
            1.0,
        )

        penalties = 0.0
        for pid in unique:
            if pid in self.user_purchased[user_id]:
                penalties += 1.0
            if pid in self.user_low_rated[user_id]:
                penalties += 0.35

        return (
            0.48 * relevance
            + 0.14 * content
            + 0.12 * collaborative
            + 0.13 * diversity
            + 0.08 * price_alignment
            + 0.05 * novelty
            - penalties
        )

    def _pick_parent(self, scored: list[tuple[list[int], float]], size: int = 3) -> list[int]:
        choices = self.rng.choice(len(scored), size=size, replace=False)
        subset = [scored[idx] for idx in choices]
        subset.sort(key=lambda item: item[1], reverse=True)
        return list(subset[0][0])

    def _crossover(self, parent_a: list[int], parent_b: list[int], candidate_ids: list[int]) -> list[int]:
        split = int(self.rng.integers(1, self.list_size))
        child = parent_a[:split]
        for gene in parent_b + candidate_ids:
            if gene not in child:
                child.append(gene)
            if len(child) == self.list_size:
                break
        return child

    def _mutate(
        self,
        chromosome: list[int],
        candidate_ids: list[int],
        weight_by_id: dict[int, float],
        mutation_rate: float,
    ) -> list[int]:
        updated = list(chromosome)

        # Apply a swap mutation and an optional replacement from the candidate pool.
        if self.rng.random() < mutation_rate and len(updated) >= 2:
            first, second = self.rng.choice(len(updated), size=2, replace=False)
            updated[first], updated[second] = updated[second], updated[first]

        if self.rng.random() >= mutation_rate:
            return updated

        pos = int(self.rng.integers(0, len(updated)))
        available = [pid for pid in candidate_ids if pid not in updated]
        if not available:
            return updated

        available_weights = np.array([weight_by_id.get(pid, 1e-6) for pid in available], dtype=float)
        if available_weights.sum() == 0:
            available_weights = np.ones_like(available_weights)
        available_weights = available_weights / available_weights.sum()
        updated[pos] = int(self.rng.choice(available, p=available_weights))
        return updated

    def _format_items(
        self,
        user_id: int,
        product_ids: list[int],
        parts: dict[str, np.ndarray],
        label: str,
    ) -> list[dict[str, Any]]:
        items = []
        for pid in product_ids:
            idx = self.product_to_idx[pid]
            items.append(
                {
                    "product_id": int(pid),
                    "name": str(self.product_meta.loc[pid, "name"]),
                    "category": str(self.product_meta.loc[pid, "category"]),
                    "price": round(float(self.product_meta.loc[pid, "price"]), 2),
                    "avg_rating": round(float(self.product_frame.loc[pid, "avg_rating"]), 2),
                    "score": round(float(parts["final"][idx]), 4),
                    "label": label,
                    "reason": self._reason(user_id, pid, parts),
                }
            )
        return items

    def _reason(self, user_id: int, product_id: int, parts: dict[str, np.ndarray]) -> str:
        idx = self.product_to_idx[product_id]
        cat = str(self.product_meta.loc[product_id, "category"])
        reasons = []
        top_cats = {item["category"] for item in self.get_user_profile_summary(user_id)["top_categories"][:2]}

        if cat in top_cats:
            reasons.append(f"Matches your strongest category interest in {cat}")
        if parts["price"][idx] >= 0.7:
            reasons.append("Fits your typical price range")
        if parts["collaborative"][idx] >= 0.45:
            reasons.append("Supported by similar users")
        if parts["rating"][idx] >= 0.75:
            reasons.append("Carries a strong average rating")
        if not reasons:
            reasons.append("Balances relevance, novelty, and category diversity")

        return ". ".join(reasons[:3]) + "."

    def _records_from_frame(self, frame: pd.DataFrame) -> list[dict[str, Any]]:
        records = []
        for row in frame.itertuples(index=False):
            records.append(
                {
                    "product_id": int(row.product_id),
                    "name": str(row.name),
                    "category": str(row.category),
                    "price": round(float(row.price), 2),
                    "avg_rating": round(float(getattr(row, "avg_rating", 3.0)), 2),
                }
            )
        return records

    @staticmethod
    def _normalize(values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=float)
        span = values.max() - values.min()
        if span == 0:
            return np.zeros_like(values)
        return (values - values.min()) / span

    @staticmethod
    def _normalize_rows(values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=float)
        min_vals = values.min(axis=1, keepdims=True)
        span = values.max(axis=1, keepdims=True) - min_vals
        span = np.where(span == 0, 1.0, span)
        return (values - min_vals) / span
