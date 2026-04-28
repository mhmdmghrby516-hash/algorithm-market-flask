const STORAGE_KEYS = {
    CART: "mm_smartshop_cart",
    RATINGS: "mm_smartshop_ratings"
};

let cart = [];
let userRatings = {};
let activeFilter = "all";
let gaSettings = {
    generations: 18,
    populationSize: 28,
    mutationRate: 0.28
};
let comparisonState = {
    guest: [],
    baseline: [],
    optimized: [],
    selectedUser: null,
    profile: null,
    candidatePool: null,
    bestFitness: null,
    gaTrace: [],
    appliedGaSettings: null
};

function isLoggedIn() {
    return document.body.dataset.loggedIn === "true";
}

function getCurrentUserId() {
    const raw = document.body.dataset.userId;
    return raw ? parseInt(raw, 10) : null;
}

function getProductImage(productId, category) {
    const seeds = {
        Toys: "toy",
        Books: "book",
        Electronics: "electronics",
        Clothes: "fashion",
        "Home Appliances": "home",
        Sports: "sports",
        Perfumes: "perfume"
    };
    const seed = seeds[category] || `product-${productId}`;
    return `https://picsum.photos/seed/${seed}-${productId}/300/200`;
}

function escapeHtml(value) {
    return String(value ?? "").replace(/[&<>"]/g, (char) => {
        if (char === "&") return "&amp;";
        if (char === "<") return "&lt;";
        if (char === ">") return "&gt;";
        return "&quot;";
    });
}

function showToast(message, type = "info") {
    let toast = document.getElementById("toast");
    if (!toast) {
        toast = document.createElement("div");
        toast.id = "toast";
        toast.className = "toast-notification";
        document.body.appendChild(toast);
    }
    toast.textContent = message;
    toast.className = `toast-notification ${type} show`;
    window.setTimeout(() => toast.classList.remove("show"), 2200);
}

function readGaSettings() {
    const generations = document.getElementById("gaGenerations");
    const populationSize = document.getElementById("gaPopulationSize");
    const mutationRate = document.getElementById("gaMutationRate");

    if (!generations || !populationSize || !mutationRate) {
        return gaSettings;
    }

    gaSettings = {
        generations: Math.min(120, Math.max(4, Number(generations.value) || 18)),
        populationSize: Math.min(120, Math.max(8, Number(populationSize.value) || 28)),
        mutationRate: Math.min(1, Math.max(0, Number(mutationRate.value) || 0.28))
    };

    generations.value = String(gaSettings.generations);
    populationSize.value = String(gaSettings.populationSize);
    mutationRate.value = gaSettings.mutationRate.toFixed(2);
    return gaSettings;
}

function buildRecommendationUrl(userId) {
    const current = readGaSettings();
    const params = new URLSearchParams({
        generations: String(current.generations),
        population_size: String(current.populationSize),
        mutation_rate: String(current.mutationRate)
    });
    return `/api/recommendations/${userId}?${params.toString()}`;
}

function loadCart() {
    const saved = localStorage.getItem(STORAGE_KEYS.CART);
    if (!saved) {
        cart = [];
        updateCartCount();
        return;
    }
    try {
        cart = JSON.parse(saved);
    } catch {
        cart = [];
    }
    updateCartCount();
}

function saveCart() {
    localStorage.setItem(STORAGE_KEYS.CART, JSON.stringify(cart));
    updateCartCount();
}

function updateCartCount() {
    const count = cart.reduce((sum, item) => sum + (item.quantity || 1), 0);
    const node = document.getElementById("cartCount");
    if (node) node.textContent = String(count);
}

async function syncCart(productId, quantity = 1, method = "POST") {
    if (!isLoggedIn()) return;
    try {
        await fetch("/api/cart", {
            method,
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ product_id: productId, quantity })
        });
    } catch (error) {
        console.error("Cart sync failed", error);
    }
}

function addToCart(product) {
    const existing = cart.find((item) => item.product_id === product.product_id);
    if (existing) {
        existing.quantity += 1;
    } else {
        cart.push({
            product_id: product.product_id,
            name: product.name,
            price: product.price,
            category: product.category,
            image: getProductImage(product.product_id, product.category),
            quantity: 1
        });
    }
    saveCart();
    syncCart(product.product_id, 1, "POST");
    showToast(`${product.name} added to cart`, "success");
}

function removeFromCart(productId) {
    cart = cart.filter((item) => item.product_id !== productId);
    saveCart();
    syncCart(productId, 0, "DELETE");
    const modal = document.getElementById("cartModal");
    if (modal?.classList.contains("show")) renderCartModal();
}

function updateQuantity(productId, nextQuantity) {
    const item = cart.find((entry) => entry.product_id === productId);
    if (!item) return;
    if (nextQuantity <= 0) {
        removeFromCart(productId);
        return;
    }
    const delta = nextQuantity - item.quantity;
    item.quantity = nextQuantity;
    saveCart();
    if (delta !== 0) {
        syncCart(productId, delta, "POST");
    }
    const modal = document.getElementById("cartModal");
    if (modal?.classList.contains("show")) renderCartModal();
}

function getCartTotal() {
    return cart.reduce((sum, item) => sum + item.price * (item.quantity || 1), 0);
}

function renderCartModal() {
    let modal = document.getElementById("cartModal");
    if (!modal) {
        modal = document.createElement("div");
        modal.id = "cartModal";
        modal.className = "cart-modal";
        modal.innerHTML = `
            <div class="cart-modal-content">
                <div class="cart-header">
                    <h3><i class="fas fa-cart-shopping"></i> Your cart</h3>
                    <button class="close-cart" type="button">&times;</button>
                </div>
                <div class="cart-items"></div>
                <div class="cart-footer">
                    <div class="cart-total">Total <span id="cartTotal">0.00</span></div>
                    <button class="checkout-btn" type="button">Checkout demo</button>
                </div>
            </div>
        `;
        document.body.appendChild(modal);
        modal.querySelector(".close-cart").addEventListener("click", () => modal.classList.remove("show"));
        modal.querySelector(".checkout-btn").addEventListener("click", () => {
            if (!cart.length) {
                showToast("Your cart is empty", "warning");
                return;
            }
            cart = [];
            saveCart();
            renderCartModal();
            modal.classList.remove("show");
            showToast("Checkout demo completed", "success");
        });
    }

    const itemsNode = modal.querySelector(".cart-items");
    if (!itemsNode) return;

    if (!cart.length) {
        itemsNode.innerHTML = `<p class="empty-cart">Your cart is empty.</p>`;
        modal.querySelector("#cartTotal").textContent = "0.00";
        modal.classList.add("show");
        return;
    }

    itemsNode.innerHTML = cart.map((item) => `
        <div class="cart-item" data-id="${item.product_id}">
            <img src="${item.image}" alt="${escapeHtml(item.name)}" class="cart-item-img">
            <div class="cart-item-details">
                <div class="cart-item-title">${escapeHtml(item.name)}</div>
                <div class="cart-item-price">$${item.price.toFixed(2)}</div>
                <div class="cart-item-category">${escapeHtml(item.category)}</div>
            </div>
            <div class="cart-item-actions">
                <button class="cart-qty-btn" type="button" data-change="-1" data-id="${item.product_id}">-</button>
                <span class="cart-qty">${item.quantity}</span>
                <button class="cart-qty-btn" type="button" data-change="1" data-id="${item.product_id}">+</button>
                <button class="cart-remove-btn" type="button" data-id="${item.product_id}">
                    <i class="fas fa-trash"></i>
                </button>
            </div>
        </div>
    `).join("");

    modal.querySelector("#cartTotal").textContent = `$${getCartTotal().toFixed(2)}`;
    itemsNode.querySelectorAll(".cart-qty-btn").forEach((button) => {
        button.addEventListener("click", () => {
            const productId = parseInt(button.dataset.id, 10);
            const change = parseInt(button.dataset.change, 10);
            const item = cart.find((entry) => entry.product_id === productId);
            if (item) updateQuantity(productId, item.quantity + change);
        });
    });
    itemsNode.querySelectorAll(".cart-remove-btn").forEach((button) => {
        button.addEventListener("click", () => removeFromCart(parseInt(button.dataset.id, 10)));
    });

    modal.classList.add("show");
}

function setupCartIcon() {
    const cartIcon = document.getElementById("cartIcon");
    if (!cartIcon) return;
    cartIcon.addEventListener("click", (event) => {
        event.preventDefault();
        renderCartModal();
    });
}

function loadRatings() {
    const saved = localStorage.getItem(STORAGE_KEYS.RATINGS);
    if (!saved) {
        userRatings = {};
        return;
    }
    try {
        userRatings = JSON.parse(saved);
    } catch {
        userRatings = {};
    }
}

function saveRatings() {
    localStorage.setItem(STORAGE_KEYS.RATINGS, JSON.stringify(userRatings));
}

async function setRating(productId, rating) {
    userRatings[productId] = rating;
    saveRatings();
    updateProductRatingDisplay(productId, rating);
    if (isLoggedIn()) {
        try {
            await fetch("/api/rate", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ product_id: productId, rating })
            });
        } catch (error) {
            console.error("Rating sync failed", error);
        }
    }
    showToast(`Saved ${rating} star rating`, "success");
}

function updateProductRatingDisplay(productId, rating) {
    document.querySelectorAll(`.rating-stars[data-product-id='${productId}']`).forEach((container) => {
        container.querySelectorAll(".star").forEach((star, index) => {
            star.classList.toggle("active", index < rating);
        });
    });
}

function renderRatingStars(productId, avgRating = 0) {
    const userRating = Number(userRatings[productId] || 0);
    let stars = `<div class="rating-stars" data-product-id="${productId}">`;
    for (let i = 1; i <= 5; i += 1) {
        const active = userRating >= i ? "active" : "";
        stars += `<button class="star ${active}" type="button" data-value="${i}" aria-label="Rate ${i} stars">★</button>`;
    }
    stars += `<span class="avg-rating">${Number(avgRating).toFixed(1)}</span>`;
    stars += `</div>`;
    return stars;
}

function bindRatingEvents() {
    document.body.addEventListener("click", (event) => {
        const button = event.target.closest(".star");
        if (!button) return;
        const wrapper = button.closest(".rating-stars");
        if (!wrapper) return;
        const productId = parseInt(wrapper.dataset.productId, 10);
        const rating = parseInt(button.dataset.value, 10);
        setRating(productId, rating);
    });
}

function buildProductCard(product, index, variant) {
    const imageUrl = getProductImage(product.product_id, product.category);
    const reasonBlock = product.reason
        ? `<p class="product-reason">${escapeHtml(product.reason)}</p>`
        : `<p class="product-reason muted-copy">Featured for browsing.</p>`;
    const scoreBlock = product.score !== undefined
        ? `<div class="product-score"><span>${variant}</span><strong>${Number(product.score).toFixed(4)}</strong></div>`
        : "";
    return `
        <article class="product-card ${variant === "Optimized" ? "optimized-card" : ""}" data-product-id="${product.product_id}" data-category="${escapeHtml(product.category)}">
            <img class="product-image" src="${imageUrl}" alt="${escapeHtml(product.name)}" loading="lazy">
            <div class="product-badge">#${index + 1}</div>
            <h3>${escapeHtml(product.name)}</h3>
            <div class="product-category"><i class="fas fa-tag"></i> ${escapeHtml(product.category)}</div>
            <div class="product-price">$${Number(product.price).toFixed(2)}</div>
            ${scoreBlock}
            <div class="product-rating">${renderRatingStars(product.product_id, product.avg_rating || 3)}</div>
            ${reasonBlock}
            <button class="add-to-cart-btn" type="button" data-product='${JSON.stringify(product).replace(/'/g, "&#39;")}'>
                <i class="fas fa-cart-plus"></i> Add to cart
            </button>
        </article>
    `;
}

function attachAddToCartHandlers(container) {
    container.querySelectorAll(".add-to-cart-btn").forEach((button) => {
        button.addEventListener("click", () => {
            const product = JSON.parse(button.dataset.product.replace(/&#39;/g, "'"));
            addToCart(product);
        });
    });
}

function renderProductCollection(products, containerId, variant) {
    const container = document.getElementById(containerId);
    if (!container) return;
    if (!products.length) {
        container.innerHTML = `<p class="no-data">No products found for the current selection.</p>`;
        return;
    }
    container.innerHTML = products.map((product, index) => buildProductCard(product, index, variant)).join("");
    attachAddToCartHandlers(container);
}

function populateCategoryFilter(products) {
    const filter = document.getElementById("categoryFilter");
    if (!filter) return;
    const categories = [...new Set(products.map((product) => product.category))].sort();
    filter.innerHTML = `<option value="all">All categories</option>` + categories
        .map((category) => `<option value="${escapeHtml(category)}">${escapeHtml(category)}</option>`)
        .join("");
    filter.value = activeFilter;
}

function applyFilter() {
    if (!isLoggedIn()) {
        const products = activeFilter === "all"
            ? comparisonState.guest
            : comparisonState.guest.filter((item) => item.category === activeFilter);
        renderProductCollection(products, "randomProductsGrid", "Featured");
        return;
    }

    const baseline = activeFilter === "all"
        ? comparisonState.baseline
        : comparisonState.baseline.filter((item) => item.category === activeFilter);
    const optimized = activeFilter === "all"
        ? comparisonState.optimized
        : comparisonState.optimized.filter((item) => item.category === activeFilter);
    renderProductCollection(baseline, "baselineGrid", "Baseline");
    renderProductCollection(optimized, "optimizedGrid", "Optimized");
}

function renderProfile(profile, userId, candidatePool) {
    const selectedUser = document.getElementById("selectedUser");
    const candidatePoolNode = document.getElementById("candidatePool");
    const prefCats = document.getElementById("prefCats");
    const avgPrice = document.getElementById("avgPrice");
    const interactionCount = document.getElementById("interactionCount");
    const purchaseCount = document.getElementById("purchaseCount");

    if (selectedUser) selectedUser.textContent = String(userId);
    if (candidatePoolNode) candidatePoolNode.textContent = candidatePool != null ? String(candidatePool) : "-";
    if (prefCats) prefCats.textContent = profile.top_categories.map((item) => item.category).join(", ");
    if (avgPrice) avgPrice.textContent = `$${Number(profile.preferred_price).toFixed(2)} (+/- $${Number(profile.price_range).toFixed(0)})`;
    if (interactionCount) interactionCount.textContent = String(profile.interactions_count);
    if (purchaseCount) purchaseCount.textContent = String(profile.purchased_count);
}

function renderGaHistory(trace, settings, bestFitness) {
    const chart = document.getElementById("gaHistoryChart");
    const bestFitnessNode = document.getElementById("gaBestFitness");
    const generationsNode = document.getElementById("gaGenerationsUsed");
    const populationNode = document.getElementById("gaPopulationUsed");
    const mutationNode = document.getElementById("gaMutationUsed");

    if (bestFitnessNode) {
        bestFitnessNode.textContent = bestFitness != null ? Number(bestFitness).toFixed(4) : "-";
    }
    if (generationsNode) {
        generationsNode.textContent = settings?.generations != null ? String(settings.generations) : "-";
    }
    if (populationNode) {
        populationNode.textContent = settings?.population_size != null ? String(settings.population_size) : "-";
    }
    if (mutationNode) {
        mutationNode.textContent = settings?.mutation_rate != null ? Number(settings.mutation_rate).toFixed(2) : "-";
    }
    if (!chart) return;

    if (!trace || !trace.length) {
        chart.innerHTML = `<p class="chart-empty">No GA trace was returned for this user.</p>`;
        return;
    }

    const maxFitness = trace.reduce((acc, point) => Math.max(acc, Number(point.best_fitness), Number(point.mean_fitness)), 0.01);
    chart.innerHTML = trace.map((point) => {
        const generation = Math.round(Number(point.generation));
        const best = Number(point.best_fitness);
        const mean = Number(point.mean_fitness);
        return `
            <div class="ga-bar-row">
                <span class="ga-gen">${generation}</span>
                <div class="ga-bar-wrap">
                    <div class="ga-bar mean" style="width:${(mean / maxFitness) * 100}%;" title="mean ${mean.toFixed(4)}"></div>
                    <div class="ga-bar best" style="width:${(best / maxFitness) * 100}%;" title="best ${best.toFixed(4)}"></div>
                </div>
            </div>
        `;
    }).join("");
}

async function loadRandomProducts() {
    try {
        const response = await fetch("/api/random_products");
        const data = await response.json();
        comparisonState.guest = data.products || [];
        populateCategoryFilter(comparisonState.guest);
        applyFilter();
    } catch (error) {
        console.error("Failed to load guest products", error);
        const container = document.getElementById("randomProductsGrid");
        if (container) container.innerHTML = `<p class="error">Failed to load featured products.</p>`;
    }
}

async function loadRecommendations(userId) {
    const baselineGrid = document.getElementById("baselineGrid");
    const optimizedGrid = document.getElementById("optimizedGrid");
    if (baselineGrid) baselineGrid.innerHTML = `<div class="loading-spinner"><i class="fas fa-spinner"></i></div>`;
    if (optimizedGrid) optimizedGrid.innerHTML = `<div class="loading-spinner"><i class="fas fa-spinner"></i></div>`;

    try {
        const response = await fetch(buildRecommendationUrl(userId));
        const data = await response.json();
        if (!response.ok) throw new Error(data.error || "Unable to load recommendations");

        comparisonState.selectedUser = userId;
        comparisonState.baseline = data.baseline || [];
        comparisonState.optimized = data.optimized || [];
        comparisonState.profile = data.user_profile || null;
        comparisonState.candidatePool = data.candidate_pool_size;
        comparisonState.bestFitness = data.best_fitness ?? null;
        comparisonState.gaTrace = data.ga_trace || [];
        comparisonState.appliedGaSettings = data.ga_settings || null;

        populateCategoryFilter([...comparisonState.baseline, ...comparisonState.optimized]);
        renderProfile(data.user_profile, userId, data.candidate_pool_size);
        renderGaHistory(comparisonState.gaTrace, comparisonState.appliedGaSettings, comparisonState.bestFitness);
        applyFilter();
    } catch (error) {
        console.error("Recommendation loading failed", error);
        if (baselineGrid) baselineGrid.innerHTML = `<p class="error">Failed to load baseline recommendations.</p>`;
        if (optimizedGrid) optimizedGrid.innerHTML = `<p class="error">Failed to load optimized recommendations.</p>`;
        renderGaHistory([], null, null);
        showToast("Could not load recommendations", "error");
    }
}

function setupFilterHandlers() {
    const filter = document.getElementById("categoryFilter");
    const reset = document.getElementById("resetFilter");
    if (filter) {
        filter.addEventListener("change", (event) => {
            activeFilter = event.target.value;
            applyFilter();
        });
    }
    if (reset) {
        reset.addEventListener("click", () => {
            activeFilter = "all";
            if (filter) filter.value = "all";
            applyFilter();
        });
    }
}

function setupQuickUsers() {
    document.querySelectorAll(".quick-user").forEach((button) => {
        button.addEventListener("click", () => {
            const userId = parseInt(button.dataset.userId, 10);
            const select = document.getElementById("user_id");
            if (select) {
                select.value = String(userId);
                select.form?.requestSubmit();
                return;
            }
            loadRecommendations(userId);
        });
    });
}

function setupGaControls() {
    readGaSettings();
    const button = document.getElementById("applyGaSettings");
    if (!button) return;
    button.addEventListener("click", () => {
        const userId = comparisonState.selectedUser || getCurrentUserId();
        if (!userId) return;
        loadRecommendations(userId);
        showToast("Re-running GA with the current settings", "info");
    });
}

function setupMobileMenu() {
    const menu = document.querySelector(".mobile-menu-btn");
    const nav = document.querySelector(".nav-links");
    const actions = document.querySelector(".header-actions");
    if (!menu || !nav) return;
    menu.addEventListener("click", () => {
        nav.classList.toggle("show");
        actions?.classList.toggle("show");
    });
}

document.addEventListener("DOMContentLoaded", () => {
    loadCart();
    loadRatings();
    setupCartIcon();
    bindRatingEvents();
    setupFilterHandlers();
    setupQuickUsers();
    setupGaControls();
    setupMobileMenu();

    if (isLoggedIn()) {
        const userId = getCurrentUserId();
        if (userId) loadRecommendations(userId);
    } else if (document.getElementById("randomProductsGrid")) {
        loadRandomProducts();
    }
});
