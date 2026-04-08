const formatNumber = (value, digits = 0) =>
  new Intl.NumberFormat("en-IN", {
    maximumFractionDigits: digits,
    minimumFractionDigits: digits,
  }).format(value ?? 0);

const formatCurrency = (value) => `INR ${formatNumber(value ?? 0, 0)}`;
const formatPct = (value, digits = 1) => `${formatNumber(value ?? 0, digits)}%`;

const state = { selectedUserId: null };

async function fetchJson(url) {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Request failed: ${response.status}`);
  }
  return response.json();
}

function renderHeroCards(metadata) {
  const cards = [
    ["2026 users scored", formatNumber(metadata.test_rows, 0)],
    ["Categories", formatNumber(metadata.categories, 0)],
    ["Expected avoidable", formatCurrency(metadata.test_expected_avoidable)],
    ["Avg recommended bucket", formatPct(metadata.avg_recommended_bucket, 1)],
  ];

  document.getElementById("hero-cards").innerHTML = cards
    .map(
      ([label, value]) => `
        <article class="hero-card">
          <span class="label">${label}</span>
          <span class="value">${value}</span>
        </article>
      `,
    )
    .join("");
}

function renderWasteAnalysis(items) {
  const maxValue = Math.max(...items.map((item) => item.expected_avoidable_amount || 0), 1);
  document.getElementById("waste-analysis").innerHTML = items
    .map((item) => {
      const width = ((item.expected_avoidable_amount || 0) / maxValue) * 100;
      return `
        <div class="bar-item">
          <strong>${item.proof_label.replaceAll("_", " ")}</strong>
          <div style="margin-top:6px; color: var(--muted);">
            Users: ${formatNumber(item.users, 0)} · Expected avoidable: ${formatCurrency(item.expected_avoidable_amount)}
          </div>
          <div class="bar-track">
            <div class="bar-fill" style="width:${width}%"></div>
          </div>
        </div>
      `;
    })
    .join("");
}

function renderSplitMetrics(items) {
  document.getElementById("split-metrics").innerHTML = items
    .map(
      (item) => `
        <div class="metric-row">
          <strong>${item.split}</strong>
          <div style="margin-top:8px; color: var(--muted); line-height: 1.7;">
            Bucket accuracy: ${formatPct(item.bucket_accuracy * 100, 2)}<br>
            Conservative rate: ${formatPct(item.conservative_rate * 100, 2)}<br>
            Avg recommended bucket: ${formatPct(item.avg_recommended_bucket, 2)}
          </div>
        </div>
      `,
    )
    .join("");
}

function renderCategoryTable(items) {
  const rows = items
    .map(
      (item) => `
        <tr>
          <td>${item.exam_fin}</td>
          <td>${formatNumber(item.users, 0)}</td>
          <td>${formatPct(item.avg_reference_discount_pct, 1)}</td>
          <td>${formatPct(item.avg_recommended_discount_pct, 1)}</td>
          <td>${formatPct(item.avg_confidence * 100, 1)}</td>
          <td>${formatCurrency(item.total_expected_avoidable_amount)}</td>
          <td>${formatPct(item.organic_share * 100, 1)}</td>
          <td>${formatPct(item.low_history_share * 100, 1)}</td>
        </tr>
      `,
    )
    .join("");

  document.getElementById("category-table").innerHTML = `
    <table>
      <thead>
        <tr>
          <th>Category</th>
          <th>Users</th>
          <th>Avg reference</th>
          <th>Avg recommended</th>
          <th>Avg confidence</th>
          <th>Expected avoidable</th>
          <th>Organic share</th>
          <th>Low-history share</th>
        </tr>
      </thead>
      <tbody>${rows}</tbody>
    </table>
  `;
}

function userRowHtml(item) {
  const proofLabel = (item.proof_label || "uncertain").replaceAll("_", " ");
  const persona = item.persona || "unknown";
  const proofClass =
    item.proof_label === "likely_unnecessary"
      ? "risk"
      : item.proof_label === "likely_necessary"
        ? "good"
        : "warn";

  return `
    <article class="user-row ${state.selectedUserId === item.userid ? "active" : ""}" data-userid="${item.userid}">
      <div class="topline">
        <strong>${item.userid}</strong>
        <span>${item.exam_fin}</span>
        <span class="pill ${proofClass}">${proofLabel}</span>
      </div>
      <div class="subline">
        <span class="pill">${persona}</span>
        <span class="pill">Rec ${formatPct(item.recommended_discount_bucket, 0)}</span>
        <span class="pill">Ref ${formatPct(item.reference_discount_pct, 1)}</span>
        <span class="pill">Conf ${formatPct(item.confidence_score * 100, 0)}</span>
        <span class="pill">Avoidable ${formatCurrency(item.expected_unnecessary_discount_amount)}</span>
      </div>
    </article>
  `;
}

function bindUserClicks() {
  document.querySelectorAll(".user-row").forEach((row) => {
    row.addEventListener("click", () => {
      loadUserDetail(row.dataset.userid);
    });
  });
}

function renderUserResults(items) {
  document.getElementById("result-count").textContent = `${items.length} users`;
  document.getElementById("user-results").innerHTML = items.map(userRowHtml).join("");
  bindUserClicks();
}

function renderUserDetail(payload) {
  const { user, category, peers } = payload;
  const proofLabel = (user.proof_label || "uncertain").replaceAll("_", " ");
  const peerList = peers
    .map(
      (peer) => `
        <tr>
          <td>${peer.userid}</td>
          <td>${peer.persona}</td>
          <td>${formatPct(peer.recommended_discount_bucket, 0)}</td>
          <td>${formatPct(peer.reference_discount_pct, 1)}</td>
          <td>${formatCurrency(peer.expected_unnecessary_discount_amount)}</td>
        </tr>
      `,
    )
    .join("");

  document.getElementById("user-detail").innerHTML = `
    <div>
      <h3>${user.userid}</h3>
      <p style="margin-top:6px; color: var(--muted);">${user.exam_fin} · ${user.persona}</p>
      <div class="detail-grid">
        <div class="detail-stat">
          <div class="label">Recommended Bucket</div>
          <div class="value">${formatPct(user.recommended_discount_bucket, 0)}</div>
        </div>
        <div class="detail-stat">
          <div class="label">Reference Discount</div>
          <div class="value">${formatPct(user.reference_discount_pct, 1)}</div>
        </div>
        <div class="detail-stat">
          <div class="label">Model Confidence</div>
          <div class="value">${formatPct(user.confidence_score * 100, 0)}</div>
        </div>
        <div class="detail-stat">
          <div class="label">Proof Confidence</div>
          <div class="value">${formatPct(user.proof_confidence_score * 100, 0)}</div>
        </div>
      </div>
      <div class="detail-body">
        <div class="meta-grid">
          <div class="meta-item">
            <div class="key">Proof label</div>
            <div class="val">${proofLabel}</div>
          </div>
          <div class="meta-item">
            <div class="key">Expected avoidable</div>
            <div class="val">${formatCurrency(user.expected_unnecessary_discount_amount)}</div>
          </div>
          <div class="meta-item">
            <div class="key">Category default</div>
            <div class="val">${formatPct(user.default_policy_pct, 1)}</div>
          </div>
          <div class="meta-item">
            <div class="key">Fallback used</div>
            <div class="val">${user.fallback_used ? "Yes" : "No"}</div>
          </div>
        </div>
        <div class="meta-item">
          <div class="key">Reason codes</div>
          <div class="val">${user.reason_codes}</div>
        </div>
        ${
          category
            ? `
              <div class="meta-item">
                <div class="key">Category context</div>
                <div class="val">
                  ${category.exam_fin} · avg recommended ${formatPct(category.avg_recommended_discount_pct, 1)} · expected avoidable ${formatCurrency(category.total_expected_avoidable_amount)}
                </div>
              </div>
            `
            : ""
        }
        <div class="meta-item">
          <div class="key">Category peers</div>
          <div class="val">
            <div class="table-shell" style="margin-top:8px;">
              <table>
                <thead>
                  <tr>
                    <th>User</th>
                    <th>Persona</th>
                    <th>Rec</th>
                    <th>Ref</th>
                    <th>Avoidable</th>
                  </tr>
                </thead>
                <tbody>${peerList}</tbody>
              </table>
            </div>
          </div>
        </div>
      </div>
    </div>
  `;
}

async function loadUserDetail(userId) {
  state.selectedUserId = userId;
  const payload = await fetchJson(`/api/user/${encodeURIComponent(userId)}`);
  renderUserDetail(payload);
  runSearch(false);
}

async function runSearch(fetchDetail = true) {
  const params = new URLSearchParams({
    query: document.getElementById("query-input").value,
    category: document.getElementById("category-select").value,
    persona: document.getElementById("persona-select").value,
    proof_label: document.getElementById("proof-select").value,
    limit: "25",
  });
  const payload = await fetchJson(`/api/users?${params.toString()}`);
  renderUserResults(payload.items);
  if (fetchDetail && payload.items.length) {
    await loadUserDetail(payload.items[0].userid);
  }
}

async function initialize() {
  const [overview, options] = await Promise.all([fetchJson("/api/overview"), fetchJson("/api/options")]);

  renderHeroCards(overview.metadata);
  renderWasteAnalysis(overview.waste_analysis);
  renderSplitMetrics(overview.split_metrics);
  renderCategoryTable(overview.top_categories);

  options.categories.forEach((category) => {
    const option = document.createElement("option");
    option.value = category;
    option.textContent = category;
    document.getElementById("category-select").appendChild(option);
  });

  options.personas.forEach((persona) => {
    const option = document.createElement("option");
    option.value = persona;
    option.textContent = persona;
    document.getElementById("persona-select").appendChild(option);
  });

  options.proof_labels.forEach((label) => {
    const option = document.createElement("option");
    option.value = label;
    option.textContent = label.replaceAll("_", " ");
    document.getElementById("proof-select").appendChild(option);
  });

  document.getElementById("search-button").addEventListener("click", () => runSearch(true));
  document.getElementById("query-input").addEventListener("keydown", (event) => {
    if (event.key === "Enter") {
      runSearch(true);
    }
  });

  renderUserResults(overview.top_users);
  if (overview.top_users.length) {
    await loadUserDetail(overview.top_users[0].userid);
  }
}

initialize().catch((error) => {
  document.body.innerHTML = `<pre style="padding:24px;">Demo failed to load: ${error.message}</pre>`;
});
