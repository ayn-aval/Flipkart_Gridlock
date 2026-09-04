/**
 * Gridlock Dashboard — Frontend Logic
 * ====================================
 * Connects to the FastAPI backend and renders:
 * - Overview stats + charts
 * - Interactive Leaflet map with clustered markers
 * - Event simulation panel (forecast + recommendations)
 * - Analytics with EDA gallery
 */

const API_BASE = window.location.origin;

// Dataset text (event descriptions, corridor names) and CV output are written by field
// operators and flow straight into template literals below. Escape everything that is
// not markup we authored ourselves.
function esc(value) {
  if (value === null || value === undefined) return '';
  return String(value)
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;').replace(/'/g, '&#39;');
}

const SEVERITY_COLOR = { High: '#ef4444', Medium: '#f59e0b', Low: '#34d399', Unknown: '#64748b' };

function fmtDuration(mins) {
  if (mins === null || mins === undefined || isNaN(mins)) return '—';
  const m = Math.round(mins);
  if (m < 60) return `${m} min`;
  const h = Math.floor(m / 60);
  const r = m % 60;
  return r ? `${h}h ${r}m` : `${h}h`;
}

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
    if (tab.dataset.tab === 'map') {
      if (!mapInitialized) {
        initMap();
        mapInitialized = true;
      } else {
        // Force Mappls canvas to recalculate its size after the container is visible
        setTimeout(() => window.dispatchEvent(new Event('resize')), 100);
      }
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

    // Auto-select first CCTV camera
    if (tab.dataset.tab === 'cctv') {
      const cctvTitle = document.getElementById('cctv-viewer-title');
      if (cctvTitle && cctvTitle.textContent === 'Select a camera feed above to monitor...') {
        const firstCamBtn = document.querySelector('.cam-select-btn');
        if (firstCamBtn) firstCamBtn.click();
      }
    }
  });
});


// ─── API Helpers ─────────────────────────────────────────────────────────────

async function apiFetch(path, options = {}) {
  try {
    const res = await fetch(`${API_BASE}${path}`, options);
    if (!res.ok) {
      // A 500 that returns an HTML error page used to throw inside the catch and
      // mask the real status.
      let detail = `HTTP ${res.status}`;
      try {
        const err = await res.json();
        if (err && err.detail) detail = err.detail;
      } catch (_) { /* non-JSON error body */ }
      throw new Error(detail);
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

    // One summary fetch shared by the overview and the dropdowns — this endpoint
    // was previously requested twice on every page load.
    const summary = await apiFetch('/events/summary');
    const distributions = await apiFetch('/events/distributions');
    await Promise.all([
      loadOverview(summary, distributions),
      loadCorridorDropdowns(summary),
      loadAnalytics(distributions),
    ]);

    // Start background poller for CCTV Autonomous Alerts
    setInterval(pollAlerts, 3000);
    
  } catch (e) {
    document.getElementById('api-status').innerHTML = '<span class="status-indicator status-offline"></span> API Connection Error';
    document.querySelector('.status-dot').style.background = '#ef4444';
    console.error('Init failed:', e);
  }
}


// ─── Overview Tab ────────────────────────────────────────────────────────────

async function loadOverview(summary, distributions) {

  // Stats cards
  document.getElementById('stat-total').textContent = summary.total_events.toLocaleString();
  document.getElementById('stat-event-driven').textContent = summary.event_driven_count.toLocaleString();
  document.getElementById('stat-high').textContent = (summary.by_severity.High || 0).toLocaleString();
  document.getElementById('stat-closures').textContent = summary.road_closure_count.toLocaleString();
  document.getElementById('stat-corridors').textContent = summary.corridors.length;
  
  const acc = summary.model_accuracy_pct;
  const base = summary.model_baseline_pct;
  const accEl = document.getElementById('stat-accuracy');
  accEl.textContent = acc != null ? acc.toFixed(1) + '%' : '—';
  // An accuracy figure without its baseline is not interpretable.
  if (acc != null && base != null) {
    accEl.title = `Majority-class baseline ${base.toFixed(1)}%`;
    const detail = accEl.parentElement && accEl.parentElement.querySelector('.stat-detail');
    if (detail) detail.textContent = `vs ${base.toFixed(0)}% baseline`;
  }

  if (summary.date_range.min && summary.date_range.max) {
    const formatDate = (dateStr) => {
      const d = new Date(dateStr);
      if (isNaN(d)) return dateStr;
      const day = d.getDate().toString().padStart(2, '0');
      const month = d.toLocaleString('en-US', { month: 'short' });
      const year = d.getFullYear();
      return `${day}-${month}-${year}`;
    };
    
    document.getElementById('stat-date-range').textContent =
      `${formatDate(summary.date_range.min)} → ${formatDate(summary.date_range.max)}`;
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
    labels: ['Low', 'Medium', 'High', 'No measured outcome'],
    values: [
      summary.by_severity.Low || 0,
      summary.by_severity.Medium || 0,
      summary.by_severity.High || 0,
      summary.by_severity.Unknown || 0,
    ],
    colors: ['#34d399', '#f59e0b', '#ef4444', '#475569'],
  });

  // Pre-aggregated server-side; this used to pull every column of all 8,057 rows.
  const hourCounts = distributions.hourly_counts;

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


// ─── Poll Autonomous Alerts ──────────────────────────────────────────────────

let lastAlertId = null;
let currentAlertData = null;

async function pollAlerts() {
  try {
    // /alerts/live no longer carries the annotated frame. It used to ship an
    // ~880 KB base64 data URI on every 3-second poll — roughly a gigabyte an hour
    // to an idle tab — even when the alert had not changed.
    const alerts = await apiFetch('/alerts/live');
    if (!alerts || !alerts.length) return;

    const latest = alerts[alerts.length - 1];
    if (latest.id === lastAlertId) return;

    lastAlertId = latest.id;
    currentAlertData = latest;

    document.getElementById('ai-alert-status').textContent = latest.cv_status || '';
    document.getElementById('ai-alert-vehicles').textContent = latest.total_vehicles ?? '—';
    document.getElementById('ai-alert-severity').textContent = latest.predicted_severity || '—';
    document.getElementById('ai-alert-duration').textContent =
      latest.predicted_duration_min != null ? Math.round(latest.predicted_duration_min) : '—';

    const badge = document.getElementById('ai-alert-badge');
    if (badge) {
      badge.className = latest.predicted_severity === 'High'
        ? 'cctv-alert-badge high-severity' : 'cctv-alert-badge';
    }
    // Image is fetched once per alert id and cached by the browser.
    document.getElementById('ai-alert-img').src =
      `${API_BASE}/alerts/live/${encodeURIComponent(latest.id)}/image`;

    const locEl = document.getElementById('ai-alert-location');
    if (locEl) locEl.textContent = latest.location || 'Unknown location';

    document.getElementById('ai-alert-banner').style.display = 'block';
  } catch (e) {
    console.warn('Failed to poll alerts', e);
  }
}

function openAlertModal() {
  if (!currentAlertData) return;
  document.getElementById('ai-alert-modal').style.display = 'flex';
  document.getElementById('modal-location').textContent = currentAlertData.location || "Unknown Location";
  document.getElementById('modal-severity').textContent = currentAlertData.predicted_severity;
  document.getElementById('modal-duration').textContent = currentAlertData.predicted_duration_min + ' min';
  
  const modalSevBox = document.getElementById('modal-severity-box');
  if (modalSevBox) {
    if (currentAlertData.predicted_severity === 'High') {
      modalSevBox.style.background = 'rgba(239, 68, 68, 0.05)';
      modalSevBox.style.borderColor = 'rgba(239, 68, 68, 0.2)';
      document.getElementById('modal-severity').style.color = 'var(--severity-high)';
    } else {
      modalSevBox.style.background = 'rgba(16, 185, 129, 0.05)';
      modalSevBox.style.borderColor = 'rgba(16, 185, 129, 0.2)';
      document.getElementById('modal-severity').style.color = 'var(--severity-low)';
    }
  }
  
  const recsUl = document.getElementById('modal-recommendations');
  if (currentAlertData.recommendations && currentAlertData.recommendations.length > 0) {
    recsUl.innerHTML = currentAlertData.recommendations
      .map(r => `<li style="margin-bottom: 8px;">${esc(r)}</li>`).join('');
  } else {
    recsUl.innerHTML = '<li>No specific recommendations generated.</li>';
  }
}

function closeAlertModal() {
  document.getElementById('ai-alert-modal').style.display = 'none';
  document.getElementById('ai-alert-banner').style.display = 'none';
}


// ─── Map Tab ─────────────────────────────────────────────────────────────────

let map = null;
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


function createMapplsMarker(point) {
  const color = SEVERITY_COLOR[point.severity_tier] || SEVERITY_COLOR.Unknown;

  const popupHtml = `
    <div class="popup-title">${esc(point.event_cause.replace(/_/g, ' '))}</div>
    <div class="popup-detail" style="color: black;">
      <strong>Corridor:</strong> ${esc(point.corridor)}<br/>
      <strong>Impact tier:</strong> <span style="color:${color}">${esc(point.severity_tier)}</span><br/>
      ${point.duration_min != null ? `<strong>Cleared in:</strong> ${esc(fmtDuration(point.duration_min))}<br/>` : ''}
      ${point.hour_of_day != null ? `<strong>Hour:</strong> ${String(point.hour_of_day).padStart(2, '0')}:00 IST<br/>` : ''}
      ${point.requires_road_closure ? '<strong style="color:#d97706;">Road closure required</strong><br/>' : ''}
    </div>
  `;

  const marker = new mappls.Marker({
    map: map,
    position: {lat: point.lat, lng: point.lon},
    popupHtml: popupHtml
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
    const limit = document.getElementById('map-filter-limit')?.value || 500;

    let url = `/hotspots/geo?limit=${limit}`;
    if (cause) url += `&event_cause=${encodeURIComponent(cause)}`;
    if (severity) url += `&severity=${encodeURIComponent(severity)}`;
    if (corridor) url += `&corridor=${encodeURIComponent(corridor)}`;

    const data = await apiFetch(url);

    // Remove existing markers
    if (currentMarkers && currentMarkers.length > 0) {
      currentMarkers.forEach(m => mappls.remove({map: map, layer: m}));
    }
    currentMarkers = [];

    // Add new markers
    if (data.points && data.points.length > 0) {
      data.points.forEach(p => {
        if (p.lon != null && p.lat != null && !isNaN(p.lon) && !isNaN(p.lat)) {
          currentMarkers.push(createMapplsMarker(p));
        }
      });
    }

    btn.innerHTML = `<i data-lucide="refresh-cw" style="width:16px;height:16px;"></i> ${data.count} events`;
    btn.title = `Displaying up to ${limit} events`;
    if (window.lucide) lucide.createIcons();
  } catch (e) {
    btn.innerHTML = '<i data-lucide="x-circle" style="width:16px;height:16px;"></i> Error';
    if (window.lucide) lucide.createIcons();
  } finally {
    btn.disabled = false;
  }
}


// ─── Populate Dropdowns ──────────────────────────────────────────────────────

async function loadCorridorDropdowns(summary) {

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

  const stationSelect = document.getElementById('sim-station');
  const zoneSelect = document.getElementById('sim-zone');
  const corridorSelect = document.getElementById('sim-corridor');
  const zoneFeedback = document.getElementById('sim-zone-feedback');
  const corridorFeedback = document.getElementById('sim-corridor-feedback');
  
  if (stationSelect && zoneSelect && summary.station_to_zone) {
    stationSelect.addEventListener('change', (e) => {
      const selectedStation = e.target.value;
      if (selectedStation) {
        if (summary.station_to_zone[selectedStation]) {
          zoneSelect.value = summary.station_to_zone[selectedStation];
          if (zoneFeedback) zoneFeedback.style.display = 'block';
        }
        if (summary.station_to_top_corridor && summary.station_to_top_corridor[selectedStation]) {
          corridorSelect.value = summary.station_to_top_corridor[selectedStation];
          if (corridorFeedback) corridorFeedback.style.display = 'block';
        }
      } else {
        if (zoneFeedback) zoneFeedback.style.display = 'none';
        if (corridorFeedback) corridorFeedback.style.display = 'none';
      }
    });
  }
  
  const vehOptions = summary.veh_types.map(c => `<option value="${c}">${c.replace(/_/g, ' ')}</option>`).join('');
  const vehEl = document.getElementById('sim-veh');
  vehEl.innerHTML = `<option value="none" selected>None / N/A</option>` + vehOptions;
}


// ─── Simulation ──────────────────────────────────────────────────────────────

// Add dynamic "specify other" inputs
document.getElementById('sim-form').addEventListener('change', (e) => {
  if (e.target.tagName === 'SELECT') {
    const isOther = e.target.value.toLowerCase() === 'others';
    let specifyInput = e.target.parentNode.querySelector('.specify-other');
    
    if (isOther) {
      if (!specifyInput) {
        specifyInput = document.createElement('input');
        specifyInput.type = 'text';
        specifyInput.className = 'form-control specify-other';
        specifyInput.style.marginTop = '8px';
        specifyInput.placeholder = 'Please specify...';
        specifyInput.dataset.forSelect = e.target.id;
        e.target.parentNode.appendChild(specifyInput);
      }
      specifyInput.style.display = 'block';
      specifyInput.required = true;
    } else if (specifyInput) {
      specifyInput.style.display = 'none';
      specifyInput.required = false;
      specifyInput.value = '';
    }
  }

  // Toggle Involved Vehicle dropdown
  if (e.target.id === 'sim-cause') {
    const cause = e.target.value;
    const vehGroup = document.getElementById('sim-veh-group');
    if (vehGroup) {
      if (cause === 'vehicle_breakdown' || cause === 'accident' || cause === 'others') {
        vehGroup.style.display = 'block';
      } else {
        vehGroup.style.display = 'none';
        document.getElementById('sim-veh').value = 'none';
      }
    }
  }
});

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

    // Helper to get value, prioritizing the "specify other" input if applicable
    const getVal = (id) => {
      const selectVal = document.getElementById(id).value;
      if (selectVal.toLowerCase() === 'others') {
        const specifyInput = document.querySelector(`.specify-other[data-for-select="${id}"]`);
        if (specifyInput && specifyInput.value.trim() !== '') {
          return specifyInput.value.trim().toLowerCase();
        }
      }
      return selectVal;
    };

    let description = document.getElementById('sim-desc').value;
    
    // Also append the custom inputs to the NLP description so the model can learn from it immediately
    let customInputs = [];
    document.querySelectorAll('#sim-form .specify-other').forEach(input => {
      if (input.style.display !== 'none' && input.value.trim()) {
        customInputs.push(input.value.trim());
      }
    });
    if (customInputs.length > 0) {
      description += description ? ' | ' : '';
      description += 'Custom params: ' + customInputs.join(', ');
    }

    const payload = {
      event_cause: getVal('sim-cause'),
      event_type: getVal('sim-type'),
      corridor: getVal('sim-corridor'),
      zone: getVal('sim-zone'),
      police_station: getVal('sim-station'),
      direction: "Unknown",
      hour_of_day: hourVal,
      day_of_week: parseInt(document.getElementById('sim-day').value),
      is_weekend: (parseInt(document.getElementById('sim-day').value) >= 5) ? 1 : 0,
      requires_road_closure: parseInt(document.getElementById('sim-closure').value),
      veh_type: getVal('sim-veh'),
      description: description,
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

  const sevColor = SEVERITY_COLOR[f.severity_tier] || SEVERITY_COLOR.Unknown;
  const confidencePct = Math.round(f.severity_confidence * 100);
  const range = f.duration_range_min || {};

  const probsHtml = ['Low', 'Medium', 'High'].map(k => `
    <div style="display:flex; flex-direction:column;">
      <span class="text-muted" style="font-size:0.7rem;">${k}</span>
      <span style="font-weight:600; color:${SEVERITY_COLOR[k]}; font-size:1.1rem;">
        ${Math.round((f.severity_probabilities[k] || 0) * 100)}%
      </span>
    </div>`).join('');

  const actionsHtml = r.action_checklist
    .map((a, i) => `<li><span class="action-num">${i + 1}</span>${esc(a)}</li>`).join('');

  const diversionsHtml = r.diversion_suggestions.map(d => `
    <div class="diversion-card">
      <span class="diversion-icon"><i data-lucide="shuffle" style="width:18px;height:18px;"></i></span>
      <div>
        <div class="diversion-name">${esc(d.corridor)}</div>
        <div class="diversion-detail">${esc(d.rationale)}</div>
      </div>
    </div>`).join('');

  const similarHtml = f.similar_past_events.slice(0, 5).map(e => `
    <div style="display:flex; justify-content:space-between; align-items:center; padding:12px 0; border-bottom:1px solid var(--border-color);">
      <div style="display:flex; flex-direction:column; gap:4px; min-width:0;">
        <div style="display:flex; align-items:center; gap:8px; flex-wrap:wrap;">
          <span style="font-weight:500; text-transform:capitalize;">${esc(e.event_cause.replace(/_/g, ' '))}</span>
          <span style="font-size:0.75rem; padding:2px 6px; background:rgba(255,255,255,0.1); border-radius:4px; color:var(--text-secondary);">${esc(e.corridor)}</span>
        </div>
        <div style="font-size:0.8rem; color:var(--text-muted);">
          Cleared in ${esc(fmtDuration(e.duration_min))} &middot;
          <span style="color:${SEVERITY_COLOR[e.severity_tier] || SEVERITY_COLOR.Unknown};">${esc(e.severity_tier)}</span>
        </div>
      </div>
      <div style="text-align:right;">
        <span style="font-size:0.75rem; color:var(--text-muted); text-transform:uppercase;">Match</span><br/>
        <span style="font-weight:700;">${Math.round(e.similarity_score * 100)}%</span>
      </div>
    </div>`).join('');

  const warningHtml = f.duration_warning ? `
    <div style="margin-top:12px; padding:10px 12px; background:rgba(245,158,11,0.08); border-left:3px solid var(--accent-orange); border-radius:4px; font-size:0.82rem; color:var(--text-secondary);">
      <strong>Wide spread:</strong> ${esc(f.duration_warning)}
    </div>` : '';

  const sparseHtml = f.sparse_cause_note ? `
    <div style="margin-top:12px; padding:10px 12px; background:rgba(59,130,246,0.08); border-left:3px solid var(--accent-blue); border-radius:4px; font-size:0.82rem; color:var(--text-secondary);">
      <strong>Thin history:</strong> ${esc(f.sparse_cause_note)}
    </div>` : '';

  const analogHtml = f.analog_duration_median_min != null ? `
    <div style="margin-top:8px; font-size:0.8rem; color:var(--text-muted);">
      Nearest-neighbour cross-check: median ${esc(fmtDuration(f.analog_duration_median_min))}
      (${esc(fmtDuration(f.analog_duration_p10_min))}–${esc(fmtDuration(f.analog_duration_p90_min))}
      across ${esc(f.analog_sample_size)} similar events) &middot;
      regression model: ${esc(fmtDuration(f.model_duration_min))}
    </div>` : '';

  container.innerHTML = `
    <div class="result-section">
      <div class="result-section-title" style="display:flex; justify-content:space-between; align-items:center;">
        <span><i data-lucide="bar-chart-2" style="width:16px;height:16px;"></i> AI Assessment</span>
        <span style="font-size:0.75rem; background:rgba(255,255,255,0.1); padding:2px 8px; border-radius:4px; font-family:monospace;">ID: ${esc(result.event_id)}</span>
      </div>

      <div class="ai-assessment-header" style="display:flex; justify-content:space-between; align-items:center; margin-bottom:16px; flex-wrap:wrap; gap:12px;">
        <div class="severity-badge ${esc(f.severity_tier.toLowerCase())}" style="border-color:${sevColor};">
          <i data-lucide="alert-triangle" style="width:16px;height:16px;"></i> ${esc(f.severity_tier)} impact tier
        </div>
        <div style="text-align:right;">
          <div class="text-muted" style="font-size:0.75rem; margin-bottom:4px;">
            Confidence ${confidencePct}%
            ${f.severity_baseline != null ? `&middot; baseline ${Math.round(f.severity_baseline * 100)}%` : ''}
          </div>
          <div class="confidence-bar" style="width:120px; height:6px; background:var(--border-color); border-radius:3px; overflow:hidden; display:inline-block;">
            <div style="width:${confidencePct}%; height:100%; background:${sevColor};"></div>
          </div>
        </div>
      </div>

      <div class="metric-grid" style="grid-template-columns:1fr 1fr; gap:16px; background:rgba(255,255,255,0.02); padding:16px; border-radius:8px; border:1px solid var(--border-color);">
        <div>
          <div class="text-muted" style="font-size:0.75rem; text-transform:uppercase; letter-spacing:0.05em; margin-bottom:4px;">Expected clearance</div>
          <div class="text-amber" style="font-size:1.5rem; font-weight:700;">${esc(fmtDuration(f.expected_duration_min))}</div>
          <div class="text-muted" style="font-size:0.78rem; margin-top:2px;">
            typical range ${esc(fmtDuration(range.p25))} – ${esc(fmtDuration(range.p75))}
          </div>
        </div>
        <div>
          <div class="text-muted" style="font-size:0.75rem; text-transform:uppercase; letter-spacing:0.05em; margin-bottom:4px;">Impact tier probabilities</div>
          <div style="display:flex; gap:16px; margin-top:8px;">${probsHtml}</div>
        </div>
      </div>

      <div class="text-muted" style="font-size:0.78rem; margin-top:10px;">
        <strong>Evidence:</strong> ${esc(f.duration_basis ? f.duration_basis.description : '')}${f.severity_basis ? ` &middot; tier from ${esc(f.severity_basis.description)}` : ''}
      </div>
      ${analogHtml}
      ${warningHtml}
      ${sparseHtml}
    </div>

    <div class="result-section">
      <div class="result-section-title"><i data-lucide="shield" style="width:16px;height:16px;"></i> Suggested Deployment Plan</div>
      <div class="metric-grid">
        <div class="metric-item">
          <span class="metric-label">Traffic police personnel</span>
          <span class="metric-value text-blue">${esc(r.manpower.officers_min)}–${esc(r.manpower.officers_max)}</span>
        </div>
        <div class="metric-item">
          <span class="metric-label">Barricades required</span>
          <span class="metric-value text-cyan">${esc(r.barricading.barricades_min)}–${esc(r.barricading.barricades_max)}</span>
        </div>
      </div>
      <div class="text-muted" style="font-size:0.8rem; margin-top:8px;"><strong>Reasoning:</strong> ${esc(r.basis)}</div>
      <div class="text-muted" style="font-size:0.75rem; margin-top:12px; padding:10px; background:rgba(255,255,255,0.03); border-radius:6px; border-left:3px solid var(--text-secondary);">
        <strong>Disclaimer:</strong> ${esc(r.disclaimer)}
      </div>
    </div>

    <div class="result-section">
      <div class="result-section-title"><i data-lucide="check-square" style="width:16px;height:16px;"></i> On-Ground Action Plan</div>
      <ul class="action-list">${actionsHtml}</ul>
    </div>

    <div class="result-section">
      <div class="result-section-title"><i data-lucide="shuffle" style="width:16px;height:16px;"></i> Suggested Traffic Diversions</div>
      <div class="diversion-list">${diversionsHtml}</div>
    </div>

    <div class="result-section">
      <div class="result-section-title"><i data-lucide="history" style="width:16px;height:16px;"></i> Past Similar Incidents</div>
      ${similarHtml}
    </div>
  `;
  if (window.lucide) lucide.createIcons();
}


// ─── Analytics Tab ───────────────────────────────────────────────────────────

async function loadAnalytics(distributions) {
  // Hotspot corridors chart
  const hotspots = await apiFetch('/hotspots?group_by=corridor');
  const top10 = hotspots.hotspots.filter(h => h.corridor !== 'Non-corridor').slice(0, 10);

  renderHorizontalBarChart('chart-hotspots', {
    labels: top10.map(h => h.corridor),
    values: top10.map(h => h.event_count),
    color: 'rgba(251, 146, 60, 0.7)',
    label: 'Events',
  });

  // Road closure rate by cause — aggregated server-side.
  const closure = distributions.closure_rate_by_cause;
  renderHorizontalBarChart('chart-closure-rate', {
    labels: closure.map(c => c.event_cause.replace(/_/g, ' ')),
    values: closure.map(c => c.rate_pct),
    color: 'rgba(251, 113, 133, 0.7)',
    label: 'Closure Rate %',
  });

  // EDA Gallery
  try {
    const eda = await apiFetch('/eda');
    const gallery = document.getElementById('eda-gallery');
    gallery.innerHTML = eda.charts.map(chart => `
      <div style="background: var(--bg-glass); border-radius: var(--radius-sm); overflow: hidden; border: 1px solid var(--border-subtle); display: flex; flex-direction: column;">
        <img src="${API_BASE}${chart.url}" alt="${chart.title}" 
             style="width: 100%; display: block; border-radius: var(--radius-sm) var(--radius-sm) 0 0;" 
             loading="lazy" />
        <div style="padding: 12px; display: flex; justify-content: space-between; align-items: center; background: rgba(0,0,0,0.2);">
          <span style="font-size: 0.85rem; color: var(--text-secondary); font-weight: 500;">
            ${chart.title}
          </span>
          <a href="${API_BASE}${chart.url}" download="${chart.title}.png" target="_blank" style="padding: 6px; border: 1px solid var(--border-subtle); border-radius: 4px; text-decoration: none; color: var(--text-primary); display: flex; align-items: center; justify-content: center; background: var(--bg-card); transition: background 0.2s;" title="Export Chart" onmouseover="this.style.background='var(--hover-bg)'" onmouseout="this.style.background='var(--bg-card)'">
            <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" x2="12" y1="15" y2="3"/></svg>
          </a>
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

    // Show the LIVE metrics computed from logged feedback. The previous version
    // fetched exactly this payload, discarded it, and displayed the frozen offline
    // test-set numbers instead — so the panel never moved no matter how much
    // feedback was submitted, while the form claimed "model metrics updated".
    const m = log.metrics;
    const maeEl = document.getElementById('stat-mae');
    const accEl = document.getElementById('stat-learning-accuracy');
    const maeDetail = document.getElementById('stat-mae-detail');
    const accDetail = document.getElementById('stat-learning-accuracy-detail');

    if (m && m.mae_duration_min != null) {
      maeEl.textContent = fmtDuration(m.mae_duration_min);
      if (maeDetail) maeDetail.textContent = `over ${m.scored_duration_events} scored event(s)`;
    } else {
      maeEl.textContent = '—';
      if (maeDetail) maeDetail.textContent = 'No scored feedback yet';
    }

    if (m && m.accuracy_severity_pct != null) {
      accEl.textContent = m.accuracy_severity_pct.toFixed(1) + '%';
      if (accDetail) accDetail.textContent = `over ${m.scored_severity_events} scored event(s)`;
    } else {
      accEl.textContent = '—';
      if (accDetail) accDetail.textContent = 'No scored feedback yet';
    }

    document.getElementById('stat-logged').textContent = m ? m.total_feedback_events : 0;

    // The offline benchmark is shown separately and labelled as such, rather than
    // being passed off as a live learning-loop metric.
    try {
      const metrics = await apiFetch('/models/metrics');
      const el = document.getElementById('offline-benchmark');
      if (el && metrics.severity && metrics.duration) {
        el.innerHTML =
          `Severity ${esc((metrics.severity.accuracy * 100).toFixed(1))}% accuracy ` +
          `vs ${esc((metrics.severity.majority_class_baseline * 100).toFixed(1))}% majority-class baseline &middot; ` +
          `duration MAE ${esc(metrics.duration.mae_min)} min ` +
          `vs ${esc(metrics.duration.baseline_mae_min)} min baseline &middot; ` +
          `P10-P90 interval covers ${esc((metrics.duration.interval_coverage_p10_p90 * 100).toFixed(0))}% of outcomes ` +
          `(held-out test split, ${esc(metrics.duration.test_size)} events)`;
      }
    } catch (_) { /* metrics endpoint optional */ }

    if (!log.count) {
      document.getElementById('learning-log-body').innerHTML =
        '<tr><td colspan="4" style="padding:16px;color:var(--text-muted);">' +
        'No feedback logged yet. Submit an outcome above to start scoring the model.</td></tr>';
      return;
    }

    const scatterData = log.entries
      .filter(e => e.predicted_duration_min != null && e.actual_duration_min != null)
      .map(e => ({ x: e.predicted_duration_min, y: e.actual_duration_min }));
    if (scatterData.length) renderScatterChart('chart-learning-scatter', { dataPoints: scatterData });

    const tbody = document.getElementById('learning-log-body');
    tbody.innerHTML = log.entries.map(e => {
      const time = new Date(e.timestamp).toLocaleTimeString();
      const match = e.predicted_severity && e.actual_severity
        && e.predicted_severity === e.actual_severity;
      const sevColor = match ? 'var(--severity-low)' : 'var(--accent-orange)';
      const mark = match ? '✓' : '✗';
      const pd = e.predicted_duration_min != null ? Math.round(e.predicted_duration_min) : '—';
      const ad = e.actual_duration_min != null ? Math.round(e.actual_duration_min) : '—';
      return `
        <tr class="learning-log-row">
          <td style="padding: 8px 10px; color: var(--text-dim);">${esc(time)}</td>
          <td style="padding: 8px 10px; font-family: monospace;">${esc(e.event_id)}</td>
          <td style="padding: 8px 10px; color: ${sevColor};">${esc(e.predicted_severity)} / ${esc(e.actual_severity)} ${mark}</td>
          <td style="padding: 8px 10px;">${esc(pd)} / ${esc(ad)} min</td>
        </tr>`;
    }).join('');

    if (window.lucide) lucide.createIcons();
  } catch (e) {
    console.warn('Learning loop fetch failed', e);
  }
}


// ─── Retraining ──────────────────────────────────────────────────────────────

window.triggerRetrain = async function () {
  const btn = document.getElementById('retrain-btn');
  const status = document.getElementById('retrain-status');
  btn.disabled = true;
  status.textContent = 'Retraining…';

  try {
    await apiFetch('/models/retrain', { method: 'POST' });
    const poll = setInterval(async () => {
      try {
        const st = await apiFetch('/models/retrain/status');
        if (!st.running) {
          clearInterval(poll);
          btn.disabled = false;
          status.textContent = st.last_result === 'success'
            ? 'Retrained — models reloaded.'
            : `Retrain ${st.last_result || 'finished'}`;
          loadLearningLoop();
          setTimeout(() => { status.textContent = ''; }, 8000);
        }
      } catch (_) { /* keep polling */ }
    }, 3000);
  } catch (err) {
    btn.disabled = false;
    status.textContent = 'Retrain failed: ' + err.message;
  }
};


// The Response Planner has no separate weekend field — `is_weekend` is derived from
// the selected day when the payload is built. A listener here used to assign to a
// `sim-weekend` element that does not exist in the markup, so every change of the Day
// dropdown threw "Cannot set properties of null" and aborted the handler.
// ─── Post-Event Feedback Submission ──────────────────────────────────────────

window.submitFeedback = async function(e) {
  e.preventDefault();
  const btn = document.getElementById('fb-submit-btn');
  const msg = document.getElementById('fb-message');
  
  btn.disabled = true;
  btn.innerHTML = '<i data-lucide="loader" class="spin" style="width:16px;height:16px;"></i> Submitting...';
  if (window.lucide) lucide.createIcons();
  
  const payload = {
    event_id: document.getElementById('fb-event-id').value.trim(),
    actual_severity: document.getElementById('fb-severity').value,
    actual_duration_min: parseFloat(document.getElementById('fb-duration').value),
    feedback_notes: ""
  };
  
  try {
    await apiFetch('/feedback', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    
    msg.textContent = "Feedback successfully logged! Model metrics updated.";
    msg.style.color = "var(--severity-low)";
    msg.style.display = "block";
    document.getElementById('feedback-form').reset();
    
    // Refresh the learning loop data instantly
    loadLearningLoop();
    
    setTimeout(() => { msg.style.display = "none"; }, 3000);
  } catch (err) {
    msg.textContent = "Error: " + err.message;
    msg.style.color = "var(--severity-high)";
    msg.style.display = "block";
  } finally {
    btn.disabled = false;
    btn.innerHTML = '<i data-lucide="upload-cloud"></i> Submit to Learning Log';
    if (window.lucide) lucide.createIcons();
  }
};

// ─── CCTV AI Logic ─────────────────────────────────────────────────────────────

// ─── CCTV Live Junction Video Wall Logic ───────────────────────────────────────

async function analyzeJunction(junctionId, junctionName, evt) {
  const viewer = document.getElementById('cctv-viewer');
  const img = document.getElementById('cctv-result-img');
  const placeholder = document.getElementById('cctv-placeholder');
  const loader = document.getElementById('cctv-loader');
  const liveBadge = document.getElementById('cctv-live-badge');
  const title = document.getElementById('cctv-viewer-title');
  
  // Highlight active button
  document.querySelectorAll('.cam-select-btn').forEach(btn => {
    btn.style.borderColor = 'var(--border)';
    btn.style.boxShadow = 'none';
  });
  // `window.event` is a Chrome-only global; in Firefox this threw before the fetch
  // and the video wall silently did nothing.
  const srcBtn = (evt && evt.currentTarget) || null;
  if (srcBtn) {
    srcBtn.style.borderColor = '#10b981';
    srcBtn.style.boxShadow = '0 0 15px rgba(16, 185, 129, 0.2)';
  }

  title.textContent = "Live Feed: " + junctionName;
  placeholder.style.display = 'none';
  img.style.display = 'none';
  liveBadge.style.display = 'none';
  loader.style.display = 'block';

  try {
    const response = await fetch(`${API_BASE}/vision/junction/${junctionId}`);
    
    if (!response.ok) throw new Error("Failed to process junction feed");
    const result = await response.json();
    
    if (result.error) {
      alert(result.error);
      throw new Error(result.error);
    }
    
    // Update Image
    img.src = result.annotated_image;
    loader.style.display = 'none';
    img.style.display = 'block';
    liveBadge.style.display = 'flex';
    
    // Update Stats
    document.getElementById('cctv-status').textContent = result.status;
    document.getElementById('cctv-status').style.color = 'var(--text-primary)';
    
    document.getElementById('cctv-severity').textContent = result.severity;
    const sevCard = document.getElementById('cctv-severity-card');
    if (result.severity === 'High') {
      sevCard.style.borderColor = 'rgba(239, 68, 68, 0.4)';
      sevCard.style.boxShadow = '0 0 15px rgba(239, 68, 68, 0.1)';
      document.getElementById('cctv-severity').style.color = 'var(--severity-high)';
    } else if (result.severity === 'Medium') {
      sevCard.style.borderColor = 'rgba(245, 158, 11, 0.4)';
      sevCard.style.boxShadow = '0 0 15px rgba(245, 158, 11, 0.1)';
      document.getElementById('cctv-severity').style.color = 'var(--severity-medium)';
    } else {
      sevCard.style.borderColor = 'rgba(16, 185, 129, 0.4)';
      sevCard.style.boxShadow = '0 0 15px rgba(16, 185, 129, 0.1)';
      document.getElementById('cctv-severity').style.color = 'var(--severity-low)';
    }
    
    document.getElementById('cctv-total').textContent = result.total_vehicles;
    
    // Update Breakdown
    const breakdownHtml = Object.entries(result.breakdown).map(([type, count]) => `
      <div style="background: rgba(255, 255, 255, 0.05); border: 1px solid var(--border-subtle); border-radius: 20px; padding: 6px 14px; font-weight: 500; text-transform: capitalize; display: flex; align-items: center; gap: 8px;">
        ${esc(type)}: <span style="font-weight: 700; color: var(--accent-cyan);">${esc(count)}</span>
      </div>
    `).join('');
    
    document.getElementById('cctv-breakdown-body').innerHTML = breakdownHtml;
    
  } catch (error) {
    console.error(error);
    placeholder.style.display = 'block';
    loader.style.display = 'none';
  }
}


// ─── Initialize ──────────────────────────────────────────────────────────────

init();
