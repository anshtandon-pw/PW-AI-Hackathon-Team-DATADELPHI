const formatNumber = (value, digits = 0) =>
  new Intl.NumberFormat("en-IN", {
    maximumFractionDigits: digits,
    minimumFractionDigits: digits,
  }).format(value ?? 0);

const formatCurrency = (value) => `INR ${formatNumber(value ?? 0, 0)}`;
const formatPct = (value, digits = 1) => `${formatNumber(value ?? 0, digits)}%`;
const safeText = (value) => (value ?? "").toString();
const state = { selectedUserId: null };

function currentSearchParams() {
  return new URLSearchParams({
    query: document.getElementById("query-input").value.trim(),
    category: document.getElementById("category-select").value,
    cluster: document.getElementById("cluster-select").value,
    lead_type: document.getElementById("lead-type-select").value,
    source_group: document.getElementById("source-group-select").value,
    limit: document.getElementById("limit-select").value,
  });
}

async function fetchJson(url) {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Request failed: ${response.status}`);
  }
  return response.json();
}

function renderHeroCards(metadata) {
  const cards = [
    ["Users scored", formatNumber(metadata.clustered_users, 0)],
    ["Avg 2026 bucket", formatPct(metadata.avg_predicted_2026_bucket, 1)],
    ["Validation accuracy", formatPct((metadata.validation_bucket_accuracy || 0) * 100, 1)],
    ["Potential saving", formatCurrency(metadata.estimated_avoidable_coupon_spend)],
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

function renderClusterSummary(items) {
  const maxValue = Math.max(...items.map((item) => item.estimated_saving_if_2026_matches_history || 0), 1);
  document.getElementById("cluster-summary").innerHTML = items
    .map((item) => {
      const width = ((item.estimated_saving_if_2026_matches_history || 0) / maxValue) * 100;
      return `
        <div class="bar-item">
          <strong>${item.coupon_cluster.replaceAll("_", " ")}</strong>
          <div style="margin-top:6px; color: var(--muted);">
            Users: ${formatNumber(item.users, 0)} | Avg 2026 bucket: ${formatPct(item.avg_predicted_bucket, 1)}
          </div>
          <div style="margin-top:6px; color: var(--muted);">
            Potential saving: ${formatCurrency(item.estimated_saving_if_2026_matches_history)}
          </div>
          <div class="bar-track">
            <div class="bar-fill" style="width:${width}%"></div>
          </div>
        </div>
      `;
    })
    .join("");
}

function renderDataQuality(items, metrics) {
  const metricCards = (metrics || [])
    .map(
      (item) => `
        <div class="metric-row">
          <strong>${safeText(item.split)} split accuracy</strong>
          <div style="margin-top:8px; color: var(--muted);">
            Bucket acc ${formatPct((item.bucket_accuracy || 0) * 100, 1)} | Bucket MAE ${formatNumber(item.bucket_mae, 2)}
          </div>
        </div>
      `,
    )
    .join("");
  const qualityCards = items
    .map(
      (item) => `
        <div class="metric-row">
          <strong>${safeText(item.metric).replace(/_/g, " ")}</strong>
          <div style="margin-top:8px; color: var(--muted);">${formatNumber(item.value, 0)}</div>
        </div>
      `,
    )
    .join("");
  document.getElementById("data-quality").innerHTML = `${metricCards}${qualityCards}`;
}

function renderPolicyTable(items) {
  const rows = items
    .map(
      (item) => `
        <tr>
          <td>${item.primary_exam_fin}</td>
          <td>${item.coupon_cluster.replaceAll("_", " ")}</td>
          <td>${formatNumber(item.policy_users, 0)}</td>
          <td>${formatPct((item.policy_coupon_share || 0) * 100, 1)}</td>
          <td>${formatPct((item.policy_organic_share || 0) * 100, 1)}</td>
          <td>${formatPct(item.recommended_base_bucket, 0)}</td>
          <td>${formatPct(item.policy_floor_pct, 0)} - ${formatPct(item.policy_cap_pct, 0)}</td>
        </tr>
      `,
    )
    .join("");

  document.getElementById("policy-table").innerHTML = `
    <table>
      <thead>
        <tr>
          <th>Category</th>
          <th>Sensitivity</th>
          <th>Users</th>
          <th>Coupon share</th>
          <th>Organic share</th>
          <th>Base bucket</th>
          <th>Allowed range</th>
        </tr>
      </thead>
      <tbody>${rows}</tbody>
    </table>
  `;
}

function userRowHtml(item) {
  return `
    <article class="user-row ${state.selectedUserId === item.userid ? "active" : ""}" data-userid="${item.userid}">
      <div class="topline">
        <strong>${item.userid}</strong>
        <span>${item.primary_exam_fin}</span>
        <span class="pill">${item.lead_type}</span>
      </div>
      <div class="subline">
        <span class="pill">${item.coupon_cluster.replaceAll("_", " ")}</span>
        <span class="pill">2026 Rec ${formatPct(item.predicted_2026_discount_bucket_pct, 0)}</span>
        <span class="pill">Hist ${formatPct(item.historical_reference_discount_bucket_pct, 0)}</span>
        <span class="pill">Conf ${formatPct((item.prediction_confidence || 0) * 100, 0)}</span>
        <span class="pill">Save ${formatCurrency(item.estimated_saving_next_order_2026)}</span>
      </div>
    </article>
  `;
}

function bindUserClicks() {
  document.querySelectorAll(".user-row").forEach((row) => {
    row.addEventListener("click", () => loadUserDetail(row.dataset.userid));
  });
}

function renderUserResults(items) {
  const total = Number(document.getElementById("result-count").dataset.total || 0);
  document.getElementById("result-count").textContent = total > items.length ? `Showing ${items.length} of ${total} users` : `${items.length} users`;
  if (!items.length) {
    document.getElementById("user-results").innerHTML = `<div class="empty-state">No users found for the current filters or userid search.</div>`;
    document.getElementById("user-detail").innerHTML = `<div class="empty-state">Try an exact userid search or clear the filters.</div>`;
    return;
  }
  document.getElementById("user-results").innerHTML = items.map(userRowHtml).join("");
  bindUserClicks();
}

function renderUserDetail(payload) {
  const { user, policy, peers } = payload;
  const peerList = peers
    .map(
      (peer) => `
        <tr>
          <td>${peer.userid}</td>
          <td>${peer.lead_type}</td>
          <td>${formatPct(peer.predicted_2026_discount_bucket_pct, 0)}</td>
          <td>${formatPct(peer.historical_reference_discount_bucket_pct, 0)}</td>
          <td>${formatCurrency(peer.estimated_saving_next_order_2026)}</td>
          <td>${formatPct((peer.prediction_confidence || 0) * 100, 0)}</td>
        </tr>
      `,
    )
    .join("");

  const actualComparison =
    user.actual_2026_discount_bucket_pct !== null && user.actual_2026_discount_bucket_pct !== undefined && user.actual_2026_discount_bucket_pct !== ""
      ? `
        <div class="meta-item">
          <div class="key">Actual 2026 comparison</div>
          <div class="val">
            Actual bucket ${formatPct(user.actual_2026_discount_bucket_pct, 0)} | Predicted bucket ${formatPct(
              user.predicted_2026_discount_bucket_pct,
              0,
            )} | Estimated saved vs actual ${formatCurrency(user.saved_amount_vs_actual_2026)}
          </div>
        </div>
      `
      : `
        <div class="meta-item">
          <div class="key">Actual 2026 comparison</div>
          <div class="val">No actual 2026 discount file has been loaded yet, so this user is being scored from 2024-2025 history only.</div>
        </div>
      `;

  document.getElementById("user-detail").innerHTML = `
    <div>
      <h3>${user.userid}</h3>
      <p style="margin-top:6px; color: var(--muted);">
        ${user.primary_exam_fin} | ${user.lead_type} | ${user.source_group}
      </p>
      <div class="detail-grid">
        <div class="detail-stat">
          <div class="label">Sensitivity Class</div>
          <div class="value" style="font-size:20px;">${user.coupon_cluster.replaceAll("_", " ")}</div>
        </div>
        <div class="detail-stat">
          <div class="label">Predicted 2026 Bucket</div>
          <div class="value">${formatPct(user.predicted_2026_discount_bucket_pct, 0)}</div>
        </div>
        <div class="detail-stat">
          <div class="label">Historical Reference</div>
          <div class="value">${formatPct(user.historical_reference_discount_bucket_pct, 0)}</div>
        </div>
        <div class="detail-stat">
          <div class="label">Confidence</div>
          <div class="value">${formatPct((user.prediction_confidence || 0) * 100, 0)}</div>
        </div>
      </div>
      <div class="detail-body">
        <div class="meta-grid">
          <div class="meta-item">
            <div class="key">Next-order saving</div>
            <div class="val">${formatCurrency(user.estimated_saving_next_order_2026)}</div>
          </div>
          <div class="meta-item">
            <div class="key">Saving if 2026 matches history</div>
            <div class="val">${formatCurrency(user.estimated_saving_if_2026_matches_history)}</div>
          </div>
          <div class="meta-item">
            <div class="key">Total orders</div>
            <div class="val">${formatNumber(user.total_orders, 0)}</div>
          </div>
          <div class="meta-item">
            <div class="key">Coupon share</div>
            <div class="val">${formatPct((user.coupon_share || 0) * 100, 1)}</div>
          </div>
          <div class="meta-item">
            <div class="key">Reference support</div>
            <div class="val">${formatPct((user.historical_reference_support_share || 0) * 100, 1)}</div>
          </div>
        </div>
        <div class="meta-item">
          <div class="key">Reason codes</div>
          <div class="val">${safeText(user.prediction_reason_codes || user.cluster_reason_codes)}</div>
        </div>
        ${
          policy
            ? `
              <div class="meta-item">
                <div class="key">Category-class policy</div>
                <div class="val">
                  Base bucket ${formatPct(policy.recommended_base_bucket, 0)} | floor ${formatPct(policy.policy_floor_pct, 0)} | cap ${formatPct(policy.policy_cap_pct, 0)} | users ${formatNumber(policy.policy_users, 0)}
                </div>
              </div>
            `
            : ""
        }
        ${actualComparison}
        <div class="meta-item">
          <div class="key">Similar users in same category + sensitivity class</div>
          <div class="val">
            <div class="table-shell" style="margin-top:8px;">
              <table>
                <thead>
                  <tr>
                    <th>User</th>
                    <th>Lead type</th>
                    <th>2026 Rec</th>
                    <th>Hist</th>
                    <th>Next-order save</th>
                    <th>Conf</th>
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
  const payload = await fetchJson(`/api/sample/user/${encodeURIComponent(userId)}`);
  renderUserDetail(payload);
  runSearch(false);
}

async function runSearch(fetchDetail = true) {
  const params = currentSearchParams();
  const payload = await fetchJson(`/api/sample/users?${params.toString()}`);
  document.getElementById("result-count").dataset.total = payload.total_matches || payload.items.length;
  renderUserResults(payload.items);
  if (fetchDetail && payload.items.length) {
    await loadUserDetail(payload.items[0].userid);
  }
}

async function initialize() {
  const [overview, options, policies] = await Promise.all([
    fetchJson("/api/sample/overview"),
    fetchJson("/api/sample/options"),
    fetchJson("/api/sample/policies?limit=20"),
  ]);

  renderHeroCards(overview.metadata);
  renderClusterSummary(overview.cluster_summary);
  renderDataQuality(overview.data_quality, overview.prediction_metrics);
  renderPolicyTable(policies.items);

  options.categories.forEach((value) => {
    const option = document.createElement("option");
    option.value = value;
    option.textContent = value;
    document.getElementById("category-select").appendChild(option);
  });
  options.clusters.forEach((value) => {
    const option = document.createElement("option");
    option.value = value;
    option.textContent = value;
    document.getElementById("cluster-select").appendChild(option);
  });
  options.lead_types.forEach((value) => {
    const option = document.createElement("option");
    option.value = value;
    option.textContent = value;
    document.getElementById("lead-type-select").appendChild(option);
  });
  options.source_groups.forEach((value) => {
    const option = document.createElement("option");
    option.value = value;
    option.textContent = value;
    document.getElementById("source-group-select").appendChild(option);
  });

  document.getElementById("search-button").addEventListener("click", () => runSearch(true));
  document.getElementById("download-button").addEventListener("click", () => {
    const params = currentSearchParams();
    window.open(`/api/sample/users-export?${params.toString().replace(/limit=[^&]+/, "")}`, "_blank");
  });
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
  document.body.innerHTML = `<pre style="padding:24px;">Sample demo failed to load: ${error.message}</pre>`;
});
