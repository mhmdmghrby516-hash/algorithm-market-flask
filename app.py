from __future__ import annotations

import json
from pathlib import Path

from flask import Flask, jsonify, redirect, render_template, request, session, url_for

from src.recommender import HybridGARecommender


app = Flask(__name__)
app.secret_key = "mm-smartshop-demo-secret"

recommender = HybridGARecommender(data_dir="data")

PROJECT_REFERENCE = {
    "title": "M&M SmartShop Hybrid Recommender",
    "method": "Hybrid baseline plus genetic optimization over top-5 product lists",
    "dataset": "users.xlsx, products.xlsx, ratings.xlsx, behavior_15500.xlsx",
}

user_carts: dict[int, dict[int, int]] = {}
user_ratings: dict[int, dict[int, int]] = {}

PRODUCT_IMAGE_LIBRARY: dict[str, list[str]] = {
    "Clothes": [
        "https://images.unsplash.com/photo-1483985988355-763728e1935b?auto=format&fit=crop&w=900&q=80",
        "https://images.unsplash.com/photo-1441984904996-e0b6ba687e04?auto=format&fit=crop&w=900&q=80",
        "https://images.unsplash.com/photo-1529139574466-a303027c1d8b?auto=format&fit=crop&w=900&q=80",
    ],
    "Electronics": [
        "https://images.unsplash.com/photo-1511707171634-5f897ff02aa9?auto=format&fit=crop&w=900&q=80",
        "https://images.unsplash.com/photo-1519389950473-47ba0277781c?auto=format&fit=crop&w=900&q=80",
        "https://images.unsplash.com/photo-1498049794561-7780e7231661?auto=format&fit=crop&w=900&q=80",
    ],
    "Perfumes": [
        "https://images.unsplash.com/photo-1541643600914-78b084683601?auto=format&fit=crop&w=900&q=80",
        "https://images.unsplash.com/photo-1594035910387-fea47794261f?auto=format&fit=crop&w=900&q=80",
        "https://images.unsplash.com/photo-1615634260167-c8cdede054de?auto=format&fit=crop&w=900&q=80",
    ],
    "Sports": [
        "https://images.unsplash.com/photo-1517836357463-d25dfeac3438?auto=format&fit=crop&w=900&q=80",
        "https://images.unsplash.com/photo-1517649763962-0c623066013b?auto=format&fit=crop&w=900&q=80",
        "https://images.unsplash.com/photo-1571019614242-c5c5dee9f50b?auto=format&fit=crop&w=900&q=80",
    ],
    "Books": [
        "https://images.unsplash.com/photo-1512820790803-83ca734da794?auto=format&fit=crop&w=900&q=80",
        "https://images.unsplash.com/photo-1495446815901-a7297e633e8d?auto=format&fit=crop&w=900&q=80",
        "https://images.unsplash.com/photo-1507842217343-583bb7270b66?auto=format&fit=crop&w=900&q=80",
    ],
    "Home Appliances": [
        "https://images.unsplash.com/photo-1586208958839-06c17cacdf08?auto=format&fit=crop&w=900&q=80",
        "https://images.unsplash.com/photo-1556911220-bff31c812dba?auto=format&fit=crop&w=900&q=80",
        "https://images.unsplash.com/photo-1600566752355-35792bedcfea?auto=format&fit=crop&w=900&q=80",
    ],
    "Toys": [
        "https://images.unsplash.com/photo-1566576912321-d58ddd7a6088?auto=format&fit=crop&w=900&q=80",
        "https://images.unsplash.com/photo-1558060370-d644479cb6f7?auto=format&fit=crop&w=900&q=80",
        "https://images.unsplash.com/photo-1587654780291-39c9404d746b?auto=format&fit=crop&w=900&q=80",
    ],
}


def product_image_url(category: str, product_id: int) -> str:
    options = PRODUCT_IMAGE_LIBRARY.get(category)
    if not options:
        options = PRODUCT_IMAGE_LIBRARY["Clothes"]
    return options[int(product_id) % len(options)]


def load_evaluation_summary() -> dict:
    path = Path("docs/evaluation_summary.json")
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def ga_defaults() -> dict[str, float | int]:
    return {
        "generations": recommender.ga_generations,
        "population_size": recommender.population_size,
        "mutation_rate": recommender.mutation_rate,
    }


def build_delivery_status() -> dict[str, object]:
    return {
        "ready_files": {
            "report": Path("docs/report_ar.md").exists(),
            "video_script": Path("docs/video_script_ar.md").exists(),
            "checklist": Path("docs/submission_checklist_ar.md").exists(),
            "render_config": Path("render.yaml").exists(),
            "wsgi": Path("wsgi.py").exists(),
        },
        "manual_items": [
            "Replace the demo secret key before public deployment.",
            "Publish the repository and update the final links in the report.",
            "Deploy the app and record a short walkthrough video.",
            "Optionally connect ratings and carts to a database for persistence.",
        ],
    }


def current_user_id() -> int | None:
    raw = session.get("user_id")
    return int(raw) if raw is not None else None


def _parse_int_query_arg(name: str, minimum: int, maximum: int) -> int | None:
    raw = request.args.get(name)
    if raw in (None, ""):
        return None
    try:
        value = int(raw)
    except ValueError:
        return None
    return max(minimum, min(maximum, value))


def _parse_float_query_arg(name: str, minimum: float, maximum: float) -> float | None:
    raw = request.args.get(name)
    if raw in (None, ""):
        return None
    try:
        value = float(raw)
    except ValueError:
        return None
    return max(minimum, min(maximum, value))


def parse_ga_settings() -> dict[str, int | float]:
    settings: dict[str, int | float] = {}
    generations = _parse_int_query_arg("generations", 4, 120)
    population_size = _parse_int_query_arg("population_size", 8, 120)
    mutation_rate = _parse_float_query_arg("mutation_rate", 0.0, 1.0)

    if generations is not None:
        settings["generations"] = generations
    if population_size is not None:
        settings["population_size"] = population_size
    if mutation_rate is not None:
        settings["mutation_rate"] = mutation_rate
    return settings


@app.context_processor
def inject_template_helpers() -> dict[str, object]:
    return {"product_image_url": product_image_url}


@app.route("/")
def index():
    user_id = current_user_id()
    initial_recommendations = (
        recommender.compare_recommendations(user_id, ga_overrides=parse_ga_settings())
        if user_id is not None
        else None
    )
    return render_template(
        "index.html",
        logged_in=user_id is not None,
        user_id=user_id,
        initial_recommendations=initial_recommendations,
        dataset_summary=recommender.dataset_summary(),
        evaluation_summary=load_evaluation_summary(),
        sample_users=recommender.sample_user_ids(),
        delivery_status=build_delivery_status(),
        project_reference=PROJECT_REFERENCE,
        ga_defaults=ga_defaults(),
    )


@app.route("/about")
def about():
    return render_template(
        "about.html",
        logged_in=current_user_id() is not None,
        user_id=current_user_id(),
        dataset_summary=recommender.dataset_summary(),
        evaluation_summary=load_evaluation_summary(),
        project_reference=PROJECT_REFERENCE,
    )


@app.route("/contact")
def contact():
    return render_template(
        "contact.html",
        logged_in=current_user_id() is not None,
        user_id=current_user_id(),
        delivery_status=build_delivery_status(),
    )


@app.route("/login", methods=["GET", "POST"])
def login():
    all_users = recommender.user_ids
    if request.method == "POST":
        user_id = request.form.get("user_id")
        if user_id and int(user_id) in recommender.user_to_idx:
            session["user_id"] = int(user_id)
            return redirect(url_for("index"))
        return render_template(
            "login.html",
            logged_in=False,
            user_id=None,
            error="Invalid user ID",
            users=all_users,
            sample_users=recommender.sample_user_ids(),
        )
    return render_template(
        "login.html",
        logged_in=False,
        user_id=None,
        users=all_users,
        sample_users=recommender.sample_user_ids(),
    )


@app.route("/logout")
def logout():
    session.pop("user_id", None)
    return redirect(url_for("index"))


@app.route("/search")
def search():
    query = request.args.get("q", "").strip()
    results = recommender.search_products(query) if query else []
    return render_template(
        "search.html",
        logged_in=current_user_id() is not None,
        user_id=current_user_id(),
        query=query,
        results=results,
    )


@app.route("/api/summary")
def summary():
    return jsonify(
        {
            "dataset": recommender.dataset_summary(),
            "evaluation": load_evaluation_summary(),
            "reference": PROJECT_REFERENCE,
            "delivery": build_delivery_status(),
            "ga_defaults": ga_defaults(),
        }
    )


@app.route("/api/health")
def health():
    return jsonify(
        {
            "status": "ok",
            "users_loaded": len(recommender.user_ids),
            "products_loaded": len(recommender.product_ids),
            "ga_defaults": ga_defaults(),
        }
    )


@app.route("/api/users")
def api_users():
    return jsonify({"users": recommender.user_ids, "sample_users": recommender.sample_user_ids()})


@app.route("/api/recommendations/<int:user_id>")
def get_recommendations(user_id: int):
    if user_id not in recommender.user_to_idx:
        return jsonify({"error": "Unknown user"}), 404

    result = recommender.compare_recommendations(user_id, ga_overrides=parse_ga_settings())
    return jsonify(
        {
            "user_id": user_id,
            "recommendations": result["optimized"],
            "baseline": result["baseline"],
            "optimized": result["optimized"],
            "user_profile": result["user_profile"],
            "candidate_pool_size": result["candidate_pool_size"],
            "best_fitness": result["best_fitness"],
            "ga_trace": result["ga_trace"],
            "ga_settings": result["ga_settings"],
        }
    )


@app.route("/api/user_profile/<int:user_id>")
def get_user_profile(user_id: int):
    if user_id not in recommender.user_to_idx:
        return jsonify({"error": "Unknown user"}), 404
    return jsonify(recommender.get_user_profile_summary(user_id))


@app.route("/api/random_products")
def random_products():
    return jsonify({"products": recommender.random_products()})


@app.route("/api/cart", methods=["GET", "POST", "DELETE"])
def cart():
    user_id = current_user_id()
    if not user_id:
        return jsonify({"error": "Not logged in"}), 401

    if request.method == "GET":
        cart_map = user_carts.get(user_id, {})
        items = []
        total = 0.0
        for pid, quantity in cart_map.items():
            if pid not in recommender.product_to_idx:
                continue
            meta = recommender.product_meta.loc[pid]
            price = float(meta["price"])
            line_total = price * quantity
            items.append(
                {
                    "product_id": int(pid),
                    "name": str(meta["name"]),
                    "category": str(meta["category"]),
                    "price": round(price, 2),
                    "quantity": int(quantity),
                    "total": round(line_total, 2),
                }
            )
            total += line_total
        return jsonify({"cart": items, "total": round(total, 2)})

    payload = request.get_json(silent=True) or {}
    product_id = payload.get("product_id")
    if not product_id:
        return jsonify({"error": "Product ID required"}), 400
    product_id = int(product_id)
    if product_id not in recommender.product_to_idx:
        return jsonify({"error": "Unknown product"}), 404

    cart_map = user_carts.setdefault(user_id, {})

    if request.method == "POST":
        quantity = int(payload.get("quantity", 1))
        cart_map[product_id] = cart_map.get(product_id, 0) + quantity
        if cart_map[product_id] <= 0:
            del cart_map[product_id]
        return jsonify({"success": True, "cart_size": sum(cart_map.values())})

    if product_id in cart_map:
        del cart_map[product_id]
    return jsonify({"success": True, "cart_size": sum(cart_map.values())})


@app.route("/api/rate", methods=["POST"])
def rate_product():
    user_id = current_user_id()
    if not user_id:
        return jsonify({"error": "Not logged in"}), 401

    payload = request.get_json(silent=True) or {}
    product_id = payload.get("product_id")
    rating = payload.get("rating")

    if product_id is None or rating is None:
        return jsonify({"error": "Product ID and rating required"}), 400

    product_id = int(product_id)
    rating = int(rating)
    if product_id not in recommender.product_to_idx:
        return jsonify({"error": "Unknown product"}), 404
    if rating < 1 or rating > 5:
        return jsonify({"error": "Rating must be between 1 and 5"}), 400

    ratings_map = user_ratings.setdefault(user_id, {})
    ratings_map[product_id] = rating
    return jsonify({"success": True, "message": "Rating saved"})


@app.route("/api/user_rating/<int:product_id>")
def get_user_rating(product_id: int):
    user_id = current_user_id()
    if not user_id:
        return jsonify({"rating": None})
    rating = user_ratings.get(user_id, {}).get(product_id)
    return jsonify({"rating": rating})


if __name__ == "__main__":
    app.run(debug=True)
