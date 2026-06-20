/**
 * Gridlock Dashboard — Frontend Logic
 * ====================================
 * Connects to the FastAPI backend and renders:
 * - Overview stats + charts
 * - Interactive Leaflet map with clustered markers
 * - Event simulation panel (forecast + recommendations)
 * - Analytics with EDA gallery
 */

const API_BASE = 'http://localhost:8000';

// ─── Tab Navigation ──────────────────────────────────────────────────────────

let mapInitialized = false;
let learningPoller = null;

document.querySelectorAll('.nav-tab').forEach(tab => {
  tab.addEventListener('click', () => {
    // Update active tab
    document.querySelectorAll('.nav-tab').forEach(t => t.classList.remove('active'));
    tab.classList.add('active');

    // Show corresponding panel
    document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
    const panelId = `panel-${tab.dataset.tab}`;
    document.getElementById(panelId).classList.add('active');

    // Lazy-init map when first shown
    if (tab.dataset.tab === 'map' && !mapInitialized) {
      initMap();
      mapInitialized = true;
    }

    // Handle learning loop polling
    if (tab.dataset.tab === 'learning') {
      loadLearningLoop();
      if (!learningPoller) {
        learningPoller = setInterval(loadLearningLoop, 3000);
      }
    } else {
      if (learningPoller) {
        clearInterval(learningPoller);
        learningPoller = null;
      }
    }
  });
});


// ─── API Helpers ─────────────────────────────────────────────────────────────

async function apiFetch(path, options = {}) {
  try {
    const res = await fetch(`${API_BASE}${path}`, options);
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || `HTTP ${res.status}`);
    }
    return await res.json();
  } catch (e) {
    console.error(`API error: ${path}`, e);
    // Update UI status to show error
    const statusDot = document.querySelector('.status-dot');
    const statusText = document.getElementById('api-status');
    if (statusDot && statusText) {
      statusDot.style.background = '#ef4444';
      statusText.textContent = 'API Connection Error';
    }
    throw e;
  }
}


// ─── Startup ─────────────────────────────────────────────────────────────────

async function init() {
  try {
    // Health check
    const health = await apiFetch('/health');
    document.getElementById('api-status').textContent =
      `${health.events_loaded.toLocaleString()} events loaded`;

    // Load all data
    await Promise.all([
      loadOverview(),
      loadCorridorDropdowns(),
      loadAnalytics(),
    ]);
  } catch (e) {
    document.getElementById('api-status').textContent = 'API Offline';
    document.querySelector('.status-dot').style.background = '#ef4444';
    console.error('Init failed:', e);
  }
}


// ─── Overview Tab ────────────────────────────────────────────────────────────

async function loadOverview() {
  const summary = await apiFetch('/events/summary');

  // Stats cards
  document.getElementById('stat-total').textContent = summary.total_events.toLocaleString();
  document.getElementById('stat-event-driven').textContent = summary.event_driven_count.toLocaleString();
  document.getElementById('stat-high').textContent = (summary.by_severity.High || 0).toLocaleString();
  document.getElementById('stat-closures').textContent = summary.road_closure_count.toLocaleString();
  document.getElementById('stat-corridors').textContent = summary.corridors.length;
  
  const acc = summary.model_accuracy_pct;
  document.getElementById('stat-accuracy').textContent = acc != null ? acc.toFixed(1) + '%' : '—';

  if (summary.date_range.min && summary.date_range.max) {
    document.getElementById('stat-date-range').textContent =
      `${summary.date_range.min} → ${summary.date_range.max}`;
  }

  // Chart: Events by Cause
  renderBarChart('chart-causes', {
    labels: Object.keys(summary.by_event_cause),
    values: Object.values(summary.by_event_cause),
    color: 'rgba(59, 130, 246, 0.8)',
    label: 'Events',
  });

  // Chart: Severity Distribution
  renderDoughnutChart('chart-severity', {
    labels: ['Low', 'High'],
    values: [summary.by_severity.Low || 0, summary.by_severity.High || 0],
    colors: ['#34d399', '#ef4444'],
  });

  // Load hourly distribution
  const events = await apiFetch('/events?limit=5000');
  const hourCounts = new Array(24).fill(0);
  events.events.forEach(e => {
    if (e.hour_of_day != null) hourCounts[Math.round(e.hour_of_day)]++;
  });

  renderBarChart('chart-hours', {
    labels: hourCounts.map((_, i) => `${String(i).padStart(2, '0')}:00`),
    values: hourCounts,
    color: 'rgba(34, 211, 238, 0.7)',
    label: 'Events',
  });

  // Chart: Top corridors
  const hotspots = await apiFetch('/hotspots?group_by=corridor');
  const top10 = hotspots.hotspots.slice(0, 10);
  renderHorizontalBarChart('chart-corridors', {
    labels: top10.map(h => h.corridor),
    values: top10.map(h => h.event_count),
    color: 'rgba(167, 139, 250, 0.7)',
    label: 'Events',
  });
}


// ─── Chart Renderers ─────────────────────────────────────────────────────────

const chartInstances = {};

function renderBarChart(canvasId, { labels, values, color, label }) {
  if (chartInstances[canvasId]) chartInstances[canvasId].destroy();

  const ctx = document.getElementById(canvasId).getContext('2d');
  chartInstances[canvasId] = new Chart(ctx, {
    type: 'bar',
    data: {
      labels,
      datasets: [{
        label,
        data: values,
        backgroundColor: color,
        borderColor: color.replace('0.8', '1').replace('0.7', '1'),
        borderWidth: 1,
        borderRadius: 4,
        maxBarThickness: 40,
      }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
      },
      scales: {
        x: {
          ticks: { color: '#94a3b8', font: { size: 10 } },
          grid: { color: 'rgba(255,255,255,0.04)' },
        },
        y: {
          ticks: { color: '#94a3b8', font: { size: 10 } },
          grid: { color: 'rgba(255,255,255,0.04)' },
        },
      },
    },
  });
}

function renderHorizontalBarChart(canvasId, { labels, values, color, label }) {
  if (chartInstances[canvasId]) chartInstances[canvasId].destroy();

  const ctx = document.getElementById(canvasId).getContext('2d');
  chartInstances[canvasId] = new Chart(ctx, {
    type: 'bar',
    data: {
      labels,
      datasets: [{
        label,
        data: values,
        backgroundColor: color,
        borderRadius: 4,
        maxBarThickness: 24,
      }],
    },
    options: {
      indexAxis: 'y',
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        x: {
          ticks: { color: '#94a3b8', font: { size: 10 } },
          grid: { color: 'rgba(255,255,255,0.04)' },
        },
        y: {
          ticks: { color: '#94a3b8', font: { size: 10 } },
          grid: { display: false },
        },
      },
    },
  });
}

function renderDoughnutChart(canvasId, { labels, values, colors }) {
  if (chartInstances[canvasId]) chartInstances[canvasId].destroy();

  const ctx = document.getElementById(canvasId).getContext('2d');
  chartInstances[canvasId] = new Chart(ctx, {
    type: 'doughnut',
    data: {
      labels,
      datasets: [{
        data: values,
        backgroundColor: colors,
        borderWidth: 0,
        hoverOffset: 8,
      }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      cutout: '65%',
      plugins: {
        legend: {
          position: 'bottom',
          labels: { color: '#94a3b8', padding: 16, font: { size: 12 } },
        },
      },
    },
  });
}

function renderScatterChart(canvasId, { dataPoints }) {
  if (chartInstances[canvasId]) chartInstances[canvasId].destroy();

  const ctx = document.getElementById(canvasId).getContext('2d');
  
  // Find max value for diagonal line
  let maxVal = 10;
  dataPoints.forEach(p => {
    if (p.x > maxVal) maxVal = p.x;
    if (p.y > maxVal) maxVal = p.y;
  });
  maxVal = Math.ceil(maxVal * 1.1);

  chartInstances[canvasId] = new Chart(ctx, {
    type: 'scatter',
    data: {
      datasets: [
        {
          label: 'Predicted vs Actual',
          data: dataPoints,
          backgroundColor: 'rgba(59, 130, 246, 0.6)',
          borderColor: 'rgba(59, 130, 246, 1)',
          pointRadius: 5,
          pointHoverRadius: 7,
        },
        {
          label: 'Ideal (x=y)',
          data: [{x: 0, y: 0}, {x: maxVal, y: maxVal}],
          type: 'line',
          borderColor: 'rgba(255, 255, 255, 0.2)',
          borderWidth: 2,
          borderDash: [5, 5],
          pointRadius: 0,
          fill: false,
        }
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        x: {
          title: { display: true, text: 'Predicted Duration (min)', color: '#94a3b8' },
          ticks: { color: '#94a3b8' },
          grid: { color: 'rgba(255,255,255,0.04)' },
          min: 0, max: maxVal
        },
        y: {
          title: { display: true, text: 'Actual Duration (min)', color: '#94a3b8' },
          ticks: { color: '#94a3b8' },
          grid: { color: 'rgba(255,255,255,0.04)' },
          min: 0, max: maxVal
        },
      },
    },
  });
}


// ─── Map Tab ─────────────────────────────────────────────────────────────────

let map;
let currentMarkers = [];
let mapplsLoaded = false;
let mapplsLoadPromise = null;

async function loadMapplsScripts() {
  if (mapplsLoadPromise) return mapplsLoadPromise;

  mapplsLoadPromise = new Promise(async (resolve, reject) => {
    try {
      const config = await apiFetch('/config');
      const apiKey = config.mappls_api_key;
      if (!apiKey) throw new Error("Mappls API key missing in .env");

      // Load main SDK
      const script1 = document.createElement('script');
      script1.src = `https://apis.mappls.com/advancedmaps/api/${apiKey}/map_sdk?layer=vector&v=3.0`;
      
      script1.onload = () => {
        // Load plugins after main SDK
        const script2 = document.createElement('script');
        script2.src = `https://apis.mappls.com/advancedmaps/api/${apiKey}/map_sdk_plugins?v=3.0`;
        script2.onload = () => {
          mapplsLoaded = true;
          resolve();
        };
        script2.onerror = reject;
        document.head.appendChild(script2);
      };
      script1.onerror = reject;
      document.head.appendChild(script1);
    } catch (e) {
      console.error("Failed to load MapMyIndia:", e);
      reject(e);
    }
  });

  return mapplsLoadPromise;
}

async function initMap() {
  const mapContainer = document.getElementById('map');
  
  try {
    mapContainer.innerHTML = '<div style="display:flex;height:100%;align-items:center;justify-content:center;color:#94a3b8;">Loading Map Engine...</div>';
    await loadMapplsScripts();
    mapContainer.innerHTML = ''; // clear loading text

    // Initialize MapMyIndia (Mappls)
    map = new mappls.Map('map', {
      center: {lat: 12.97, lng: 77.59},
      zoom: 12,
      zoomControl: true,
    });

    // Fetch and populate markers
    refreshMap();
  } catch(e) {
    mapContainer.innerHTML = '<div style="display:flex;height:100%;align-items:center;justify-content:center;color:#ef4444;">Failed to load map. Check .env API key.</div>';
  }
}

const SEVERITY_COLORS = {
  High: '#ef4444',
  Low: '#34d399',
};

function createMapplsMarker(point) {
  const color = SEVERITY_COLORS[point.severity_tier] || '#60a5fa';
  
  const popupHtml = `
    <div class="popup-title">${point.event_cause.replace(/_/g, ' ')}</div>
    <div class="popup-detail" style="color: black;">
      <strong>Corridor:</strong> ${point.corridor}<br/>
      <strong>Severity:</strong> <span style="color:${color}">${point.severity_tier}</span><br/>
      ${point.hour_of_day != null ? `<strong>Hour:</strong> ${String(point.hour_of_day).padStart(2, '0')}:00 IST<br/>` : ''}
      ${point.requires_road_closure ? '<strong style="display:flex;align-items:center;gap:4px;color:#d97706;"><i data-lucide="alert-triangle" style="width:14px;height:14px;"></i> Road Closure Required</strong><br/>' : ''}
    </div>
  `;

  // Use standard Mappls marker with clusters enabled
  const marker = new mappls.Marker({
    map: map,
    position: {lat: point.lat, lng: point.lon},
    popupHtml: popupHtml,
    clusters: true,
    clustersOptions: {
      color: "white",
      bgcolor: color
    }
  });

  return marker;
}

async function refreshMap() {
  if (!map) return;
  const btn = document.getElementById('map-refresh-btn');
  btn.disabled = true;
  btn.innerHTML = '<i data-lucide="loader" class="spin" style="width:16px;height:16px;"></i> Loading...';
  if (window.lucide) lucide.createIcons();

  try {
    const cause = document.getElementById('map-filter-cause').value;
    const severity = document.getElementById('map-filter-severity').value;
    const corridor = document.getElementById('map-filter-corridor').value;

    let url = '/hotspots/geo?limit=5000';
    if (cause) url += `&event_cause=${encodeURIComponent(cause)}`;
    if (severity) url += `&severity=${encodeURIComponent(severity)}`;
    if (corridor) url += `&corridor=${encodeURIComponent(corridor)}`;

    const data = await apiFetch(url);

    // Remove existing markers
    currentMarkers.forEach(m => mappls.remove({map: map, layer: m}));
    currentMarkers = [];

    // Add new markers
    data.points.forEach(p => {
      currentMarkers.push(createMapplsMarker(p));
    });

    btn.innerHTML = `<i data-lucide="refresh-cw" style="width:16px;height:16px;"></i> ${data.count} events`;
    if (window.lucide) lucide.createIcons();
  } catch (e) {
    btn.innerHTML = '<i data-lucide="x-circle" style="width:16px;height:16px;"></i> Error';
    if (window.lucide) lucide.createIcons();
  } finally {
    btn.disabled = false;
  }
}


// ─── Populate Dropdowns ──────────────────────────────────────────────────────

async function loadCorridorDropdowns() {
  const summary = await apiFetch('/events/summary');

  // Corridor dropdowns
  const corridors = summary.corridors;
  ['sim-corridor', 'map-filter-corridor'].forEach(id => {
    const select = document.getElementById(id);
    if (!select) return;
    const firstOption = id === 'sim-corridor' ? '<option value="Non-corridor">Non-corridor</option>' : '<option value="">All Corridors</option>';
    select.innerHTML = firstOption + corridors.map(c =>
      `<option value="${c}">${c}</option>`
    ).join('');
  });

  // Event cause dropdown for map and sim
  const causeSelectMap = document.getElementById('map-filter-cause');
  const causeSelectSim = document.getElementById('sim-cause');
  const causes = summary.event_causes;
  
  const causeOptions = causes.map(c => `<option value="${c}">${c.replace(/_/g, ' ')}</option>`).join('');
  if (causeSelectMap) causeSelectMap.innerHTML = '<option value="">All Causes</option>' + causeOptions;
  if (causeSelectSim) causeSelectSim.innerHTML = '<option value="">Select cause...</option>' + causeOptions;

  // New Dropdowns
  const populate = (id, list, defaultText) => {
    const el = document.getElementById(id);
    if (el) el.innerHTML = `<option value="">${defaultText}</option>` + list.map(i => `<option value="${i}">${i}</option>`).join('');
  };

  populate('sim-type', summary.event_types, 'Select type...');
  populate('sim-zone', summary.zones, 'Select zone...');
  populate('sim-station', summary.police_stations, 'Select station...');

  
  const vehOptions = summary.veh_types.map(c => `<option value="${c}">${c.replace(/_/g, ' ')}</option>`).join('');
  const vehEl = document.getElementById('sim-veh');
  vehEl.innerHTML = `<option value="All" selected>All / Any</option>` + vehOptions;
}


// ─── Simulation ──────────────────────────────────────────────────────────────

async function runSimulation(e) {
  e.preventDefault();

  const btn = document.getElementById('sim-submit-btn');
  btn.disabled = true;
  btn.innerHTML = '<i data-lucide="loader" class="spin" style="width:16px;height:16px;"></i> Forecasting...';
  if (window.lucide) lucide.createIcons();

  try {
    let hourVal = parseInt(document.getElementById('sim-hour').value);
    const ampm = document.getElementById('sim-ampm').value;
    if (ampm === 'PM' && hourVal < 12) hourVal += 12;
    if (ampm === 'AM' && hourVal === 12) hourVal = 0;

    const payload = {
      event_cause: document.getElementById('sim-cause').value,
      event_type: document.getElementById('sim-type').value,
      corridor: document.getElementById('sim-corridor').value,
      zone: document.getElementById('sim-zone').value,
      police_station: document.getElementById('sim-station').value,
      direction: "Unknown",
      hour_of_day: hourVal,
      day_of_week: parseInt(document.getElementById('sim-day').value),
      is_weekend: parseInt(document.getElementById('sim-weekend').value),
      requires_road_closure: parseInt(document.getElementById('sim-closure').value),
      veh_type: document.getElementById('sim-veh').value,
      description: document.getElementById('sim-desc').value,
    };

    const result = await apiFetch('/forecast', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });

    renderForecastResults(result);
  } catch (err) {
    document.getElementById('forecast-results').innerHTML = `
      <div class="empty-state">
        <h3 class="text-rose" style="display:flex;align-items:center;justify-content:center;gap:8px;"><i data-lucide="x-circle"></i> Forecast Error</h3>
        <p class="text-muted mt-4">${err.message}</p>
      </div>
    `;
    if (window.lucide) lucide.createIcons();
  } finally {
    btn.disabled = false;
    btn.innerHTML = '<i data-lucide="play" style="width:16px;height:16px;"></i> Run Forecast';
    if (window.lucide) lucide.createIcons();
  }
}

function renderForecastResults(result) {
  const f = result.forecast;
  const r = result.recommendation;
  const container = document.getElementById('forecast-results');

  const severityClass = f.severity_tier.toLowerCase();
  const confidencePct = Math.round(f.severity_confidence * 100);
  const confColor = f.severity_tier === 'High' ? '#ef4444' : '#34d399';

  let hrs = Math.floor(f.expected_duration_min / 60);
  let mins = Math.round(f.expected_duration_min % 60);
  let durationDisplay = hrs > 0 ? `${hrs} hours ${mins} mins` : `${mins} mins`;
  if (f.analog_duration_median_min) {
    let aHrs = Math.floor(f.analog_duration_median_min / 60);
    let aMins = Math.round(f.analog_duration_median_min % 60);
    let aDisplay = aHrs > 0 ? `${aHrs} hours ${aMins} mins` : `${aMins} mins`;
    durationDisplay += ` <span class="text-muted">(analog median: ${aDisplay})</span>`;
  }

  // Build actions HTML
  const actionsHtml = r.action_checklist.map((a, i) =>
    `<li><span class="action-num">${i + 1}</span>${a}</li>`
  ).join('');

  // Build diversions HTML
  const diversionsHtml = r.diversion_suggestions.map(d => `
    <div class="diversion-card">
      <span class="diversion-icon"><i data-lucide="shuffle" style="width:18px;height:18px;"></i></span>
      <div>
        <div class="diversion-name">${d.corridor}</div>
        <div class="diversion-detail">${d.rationale}</div>
      </div>
    </div>
  `).join('');

  // Build similar events HTML
  const similarHtml = f.similar_past_events.slice(0, 5).map(e => `
    <div class="similar-event">
      <div>
        <span class="cause-tag">${e.event_cause.replace(/_/g, ' ')}</span>
        <span class="text-muted" style="margin-left:6px">${e.corridor}</span>
      </div>
      <span class="duration">${e.duration_min}m</span>
      <span class="similarity">${Math.round(e.similarity_score * 100)}%</span>
    </div>
  `).join('');

  container.innerHTML = `
    <!-- Severity & Duration -->
    <div class="result-section">
      <div class="result-section-title"><i data-lucide="bar-chart-2" style="width:16px;height:16px;"></i> Forecast Result</div>
      <div style="display: flex; align-items: center; gap: 16px; flex-wrap: wrap;">
        <div class="severity-badge ${severityClass}">
          <i data-lucide="alert-circle" style="width:16px;height:16px;"></i> ${f.severity_tier} Severity
        </div>
        <span class="text-muted" style="font-size: 0.8rem;">
          Method: <strong>${f.method === 'knn_analog_fallback' ? 'k-NN Analog Fallback' : 'GBT Model'}</strong>
        </span>
      </div>
      <div class="confidence-bar" style="margin-top: 12px;">
        <div class="confidence-fill" style="width: ${confidencePct}%; background: ${confColor};"></div>
      </div>
      <div class="text-muted" style="font-size: 0.75rem; margin-top: 4px;">Confidence: ${confidencePct}%</div>
      <div class="metric-grid" style="margin-top: 16px;">
        <div class="metric-item">
          <span class="metric-label">Expected Clearance Time</span>
          <span class="metric-value text-amber">${durationDisplay}</span>
        </div>
        <div class="metric-item">
          <span class="metric-label">Severity Probabilities</span>
          <div class="sim-prob-bar" style="font-size: 0.8rem; color: var(--text-secondary);">
            Low: ${Math.round(f.severity_probabilities.Low * 100)}% · 
            High: ${Math.round(f.severity_probabilities.High * 100)}%
          </div>
        </div>
      </div>
    </div>

    <!-- Resource Recommendation -->
    <div class="result-section">
      <div class="result-section-title"><i data-lucide="shield" style="width:16px;height:16px;"></i> Suggested Deployment Plan</div>
      <div class="metric-grid">
        <div class="metric-item">
          <span class="metric-label">Traffic Police Personnel Needed</span>
          <span class="metric-value text-blue">${r.manpower.officers_min}–${r.manpower.officers_max}</span>
        </div>
        <div class="metric-item">
          <span class="metric-label">Barricades Required</span>
          <span class="metric-value text-cyan">${r.barricading.barricades_min}–${r.barricading.barricades_max}</span>
        </div>
      </div>
      <div class="text-muted" style="font-size: 0.8rem; margin-top: 8px;">
        <strong>Reasoning:</strong> ${r.basis}
      </div>
    </div>

    <!-- Action Checklist -->
    <div class="result-section">
      <div class="result-section-title"><i data-lucide="check-square" style="width:16px;height:16px;"></i> On-Ground Action Plan</div>
      <ul class="action-list">${actionsHtml}</ul>
    </div>

    <!-- Diversion Suggestions -->
    <div class="result-section">
      <div class="result-section-title"><i data-lucide="shuffle" style="width:16px;height:16px;"></i> Suggested Traffic Diversions</div>
      <div class="diversion-list">${diversionsHtml}</div>
    </div>

    <!-- Similar Past Events -->
    <div class="result-section">
      <div class="result-section-title"><i data-lucide="history" style="width:16px;height:16px;"></i> Past Similar Incidents</div>
      ${similarHtml}
    </div>
  `;
  if (window.lucide) lucide.createIcons();
}


// ─── Analytics Tab ───────────────────────────────────────────────────────────

async function loadAnalytics() {
  // Hotspot corridors chart
  const hotspots = await apiFetch('/hotspots?group_by=corridor');
  const top10 = hotspots.hotspots.filter(h => h.corridor !== 'Non-corridor').slice(0, 10);

  renderHorizontalBarChart('chart-hotspots', {
    labels: top10.map(h => h.corridor),
    values: top10.map(h => h.event_count),
    color: 'rgba(251, 146, 60, 0.7)',
    label: 'Events',
  });

  // Road closure rate by cause
  const events = await apiFetch('/events?limit=5000');
  const causeCounts = {};
  const causeClosures = {};
  events.events.forEach(e => {
    causeCounts[e.event_cause] = (causeCounts[e.event_cause] || 0) + 1;
    if (e.requires_road_closure) {
      causeClosures[e.event_cause] = (causeClosures[e.event_cause] || 0) + 1;
    }
  });

  const causes = Object.keys(causeCounts).sort((a, b) => {
    const rateA = (causeClosures[a] || 0) / causeCounts[a];
    const rateB = (causeClosures[b] || 0) / causeCounts[b];
    return rateB - rateA;
  });

  renderHorizontalBarChart('chart-closure-rate', {
    labels: causes.map(c => c.replace(/_/g, ' ')),
    values: causes.map(c => Math.round(((causeClosures[c] || 0) / causeCounts[c]) * 100)),
    color: 'rgba(251, 113, 133, 0.7)',
    label: 'Closure Rate %',
  });

  // EDA Gallery
  try {
    const eda = await apiFetch('/eda');
    const gallery = document.getElementById('eda-gallery');
    gallery.innerHTML = eda.charts.map(chart => `
      <div style="background: var(--bg-glass); border-radius: var(--radius-sm); overflow: hidden; border: 1px solid var(--border-subtle);">
        <img src="${API_BASE}${chart.url}" alt="${chart.title}" 
             style="width: 100%; display: block; border-radius: var(--radius-sm);" 
             loading="lazy" />
        <div style="padding: 8px 12px; font-size: 0.8rem; color: var(--text-secondary);">
          ${chart.title}
        </div>
      </div>
    `).join('');
  } catch (e) {
    console.warn('EDA charts not available');
  }
}

// ─── Learning Loop Tab ───────────────────────────────────────────────────────

async function loadLearningLoop() {
  try {
    const log = await apiFetch('/feedback/log?limit=50');
    
    if (log.count === 0) return;

    // Update Stats
    document.getElementById('stat-mae').textContent = 
      log.metrics.mae_duration_min !== null ? log.metrics.mae_duration_min.toFixed(1) : '—';
    document.getElementById('stat-learning-accuracy').textContent = 
      log.metrics.accuracy_severity_pct !== null ? log.metrics.accuracy_severity_pct.toFixed(1) + '%' : '—';
    document.getElementById('stat-logged').textContent = log.metrics.total_feedback_events;

    // Update Scatter Chart
    const scatterData = log.entries.map(e => ({
      x: e.predicted_duration_min || 0,
      y: e.actual_duration_min || 0,
    }));
    renderScatterChart('chart-learning-scatter', { dataPoints: scatterData });

    // Update Live Log Table
    const tbody = document.getElementById('learning-log-body');
    tbody.innerHTML = log.entries.map(e => {
      const time = new Date(e.timestamp).toLocaleTimeString();
      const sevColor = e.predicted_severity === e.actual_severity ? 'var(--text-primary)' : 'var(--accent-orange)';
      const sevMatch = e.predicted_severity === e.actual_severity ? '<i data-lucide="check" style="width:14px;height:14px;"></i>' : '<i data-lucide="x" style="width:14px;height:14px;"></i>';
      
      return `
        <tr class="learning-log-row">
          <td style="padding: 8px 10px; color: var(--text-dim);">${time}</td>
          <td style="padding: 8px 10px; font-family: monospace;">${e.event_id}</td>
          <td style="padding: 8px 10px; color: ${sevColor}; display:flex; align-items:center; gap:4px;">${e.predicted_severity} / ${e.actual_severity} ${sevMatch}</td>
          <td style="padding: 8px 10px;">${e.predicted_duration_min} / ${e.actual_duration_min}</td>
        </tr>
      `;
    }).join('');

    if (window.lucide) lucide.createIcons();

  } catch (e) {
    console.warn('Learning loop fetch failed', e);
  }
}


// ─── Auto-sync weekend field with day selection ──────────────────────────────

document.getElementById('sim-day').addEventListener('change', function () {
  const day = parseInt(this.value);
  document.getElementById('sim-weekend').value = (day >= 5) ? '1' : '0';
});


// ─── Initialize ──────────────────────────────────────────────────────────────

init();
