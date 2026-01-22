// Config
const API_URL = "http://localhost:8000";
const COLOR_ACCENT =
  getComputedStyle(document.documentElement)
    .getPropertyValue("--accent")
    .trim() || "#238636";

const DEFAULT_YEAR = 2024;

const SLIDER_CONFIG = {
  Mining: {
    feature: "Miner\u00eda",
    unit: "ha",
    min: -2000,
    max: 2000,
    step: 50,
    decimals: 0,
  },
  Agriculture: {
    feature: "area_agropec",
    unit: "ha",
    min: -20000,
    max: 20000,
    step: 500,
    decimals: 0,
  },
  Infrastructure: {
    feature: "Infraestructura",
    unit: "index",
    min: -500,
    max: 500,
    step: 10,
    decimals: 0,
  },
  Climate: {
    feature: "tmean",
    unit: "deg C",
    min: -2,
    max: 2,
    step: 0.1,
    decimals: 1,
  },
  Socioeconomic: {
    feature: "Poblaci\u00f3n",
    unit: "persons",
    min: -20000,
    max: 20000,
    step: 500,
    decimals: 0,
  },
};

const FEATURE_LABELS = {
  "Miner\u00eda": { label: "Mineria", unit: "ha", decimals: 0 },
  area_agropec: { label: "Agricultura", unit: "ha", decimals: 0 },
  Infraestructura: { label: "Infraestructura", unit: "index", decimals: 0 },
  tmean: { label: "Clima", unit: "deg C", decimals: 1 },
  "Poblaci\u00f3n": { label: "Poblacion", unit: "persons", decimals: 0 },
  Poblacion: { label: "Poblacion", unit: "persons", decimals: 0 },
};

const MAP_SOURCES = {
  department: {
    url: "data/peru_departments.geojson",
    nameProp: "NOMBDEP",
  },
  province: {
    url: "data/peru_provinces.geojson",
    nameProp: "NOMBPROB",
  },
};

// State
let bauTotal = 0;
let currentTotal = 0;
let debounceTimer = null;
let marginalResults = null;
let departmentOptions = [];
let provinceOptions = [];
let mapSvg = null;
let mapG = null;
let mapPath = null;
let mapGeoCache = {};
let mapLastLevel = null;
let mapLastValues = null;

// Charts
let contribChart = null;
let deptChart = null;

// UI Elements
const totalEl = document.getElementById("total-ha");
const bauEl = document.getElementById("bau-ha");
const deltaEl = document.getElementById("total-delta");
const yearLabelEl = document.getElementById("year-label");
const scenarioMetaEl = document.getElementById("scenario-meta");
const contribTitleEl = document.getElementById("contrib-title");
const contribSubtitleEl = document.getElementById("contrib-subtitle");
const contribEmptyEl = document.getElementById("contrib-empty");
const regionDescEl = document.getElementById("region-desc");
const mapContainer = document.getElementById("map-container");
const mapPlaceholder = document.getElementById("map-placeholder");
const mapLegend = document.getElementById("map-legend");
const mapTooltip = document.getElementById("map-tooltip");
const sidebarToggle = document.getElementById("sidebar-toggle");
const sidebarOverlay = document.getElementById("sidebar-overlay");

const sliders = document.querySelectorAll("input[type=range]");
const resetBtn = document.getElementById("reset-btn");
const modeSelect = document.getElementById("mode-select");
const yearInput = document.getElementById("year-input");
const levelSelect = document.getElementById("level-select");
const groupSelect = document.getElementById("group-select");
const groupLabelEl = document.getElementById("group-label");

function formatSignedNumber(value, decimals = 0) {
  const num = Number(value);
  if (!Number.isFinite(num)) {
    return "0";
  }
  const normalized = Math.abs(num) < 1e-9 ? 0 : num;
  const text =
    decimals > 0
      ? normalized.toFixed(decimals)
      : Math.round(normalized).toLocaleString("en-US");
  return normalized > 0 ? `+${text}` : text;
}

function formatWithUnit(value, unit, decimals = 0) {
  const base = formatSignedNumber(value, decimals);
  return unit ? `${base} ${unit}` : base;
}

function setSlidersEnabled(enabled) {
  sliders.forEach((slider) => {
    slider.disabled = !enabled;
    const group = slider.closest(".slider-group");
    if (group) {
      group.classList.toggle("is-disabled", !enabled);
    }
  });
  resetBtn.disabled = !enabled;
}

function initSliderUI() {
  sliders.forEach((slider) => {
    const config = SLIDER_CONFIG[slider.dataset.driver];
    if (!config) {
      slider.disabled = true;
      slider.closest(".slider-group")?.classList.add("is-disabled");
      return;
    }
    slider.min = config.min;
    slider.max = config.max;
    slider.step = config.step;
    slider.value = 0;
    updateSliderLabel(slider);
  });
}

function updateSliderLabel(slider) {
  const config = SLIDER_CONFIG[slider.dataset.driver];
  if (!config) {
    return;
  }
  const valueLabel = slider.parentElement.querySelector(".value");
  if (!valueLabel) {
    return;
  }
  const numeric = Number(slider.value);
  valueLabel.innerText = formatWithUnit(numeric, config.unit, config.decimals);
}

function getScenario() {
  const mode = modeSelect?.value || "hindcast";
  let year = Number.parseInt(yearInput?.value, 10);
  if (!Number.isFinite(year)) {
    year = DEFAULT_YEAR;
  }
  const min = Number.parseInt(yearInput?.min || "1900", 10);
  const max = Number.parseInt(yearInput?.max || "2100", 10);
  year = Math.min(max, Math.max(min, year));
  if (yearInput) {
    yearInput.value = String(year);
  }
  return { mode, year };
}

function getGroupBy() {
  return levelSelect?.value === "province" ? "province" : "department";
}

function syncScenarioMeta() {
  const { mode, year } = getScenario();
  if (scenarioMetaEl) {
    scenarioMetaEl.textContent = `Modo: ${mode} | Ano: ${year}`;
  }
  if (yearLabelEl) {
    yearLabelEl.textContent = year;
  }
}

function buildFeatureDeltas() {
  const deltas = {};
  sliders.forEach((slider) => {
    const config = SLIDER_CONFIG[slider.dataset.driver];
    if (!config) {
      return;
    }
    const value = Number(slider.value);
    if (!Number.isFinite(value) || value === 0) {
      return;
    }
    deltas[config.feature] = value;
  });
  return deltas;
}

async function fetchDepartments() {
  const res = await fetch(`${API_URL}/meta/departments`);
  if (!res.ok) {
    throw new Error(`API error (${res.status})`);
  }
  const data = await res.json();
  return Array.isArray(data.departments) ? data.departments : [];
}

async function fetchProvinces() {
  const res = await fetch(`${API_URL}/meta/provinces`);
  if (!res.ok) {
    throw new Error(`API error (${res.status})`);
  }
  const data = await res.json();
  return Array.isArray(data.provinces) ? data.provinces : [];
}

function populateGroupSelect(options) {
  const current = groupSelect.value;
  groupSelect.innerHTML = "";
  if (!options.length) {
    const opt = document.createElement("option");
    opt.value = "";
    opt.textContent = "Sin datos";
    groupSelect.appendChild(opt);
    groupSelect.disabled = true;
    return "";
  }

  options.forEach((name) => {
    const opt = document.createElement("option");
    opt.value = name;
    opt.textContent = name;
    groupSelect.appendChild(opt);
  });

  const selected = options.includes(current) ? current : options[0];
  groupSelect.value = selected;
  groupSelect.disabled = false;
  return selected;
}

async function loadGroupOptions() {
  updateGroupLabel();
  groupSelect.disabled = true;
  groupSelect.innerHTML = `<option value="">Cargando...</option>`;

  try {
    if (levelSelect.value === "province") {
      if (!provinceOptions.length) {
        provinceOptions = await fetchProvinces();
      }
      populateGroupSelect(provinceOptions);
    } else {
      if (!departmentOptions.length) {
        departmentOptions = await fetchDepartments();
      }
      populateGroupSelect(departmentOptions);
    }
  } catch (err) {
    console.error("Failed to load groups", err);
    populateGroupSelect([]);
  }
}

function normalizeKey(value) {
  return String(value || "")
    .normalize("NFD")
    .replace(/[\u0300-\u036f]/g, "")
    .trim()
    .toUpperCase();
}

function setMapPlaceholder(message) {
  if (!mapPlaceholder) {
    return;
  }
  mapPlaceholder.textContent = message;
  mapPlaceholder.style.display = message ? "block" : "none";
  if (message && mapLegend) {
    mapLegend.innerHTML = "";
  }
}

function ensureMapSvg() {
  if (!mapContainer) {
    return;
  }
  if (!mapSvg) {
    mapSvg = window.d3
      .select(mapContainer)
      .append("svg")
      .attr("class", "choropleth");
    mapG = mapSvg.append("g");
  }
}

function mapSize() {
  const rect = mapContainer.getBoundingClientRect();
  const width = Math.max(300, Math.floor(rect.width));
  const height = Math.max(360, Math.floor(rect.height));
  return { width, height };
}

async function loadGeoData(level) {
  const config = MAP_SOURCES[level];
  if (!config) {
    return null;
  }
  if (mapGeoCache[level]) {
    return mapGeoCache[level];
  }
  if (!window.d3?.json) {
    throw new Error("D3 not loaded");
  }
  const data = await window.d3.json(config.url);
  mapGeoCache[level] = data;
  return data;
}

function resolveNameProperty(features, preferred) {
  if (!features || !features.length) {
    return preferred;
  }
  if (features[0].properties && preferred in features[0].properties) {
    return preferred;
  }
  const fallback = ["NOMBDEP", "NOMBPROB", "name", "NAME"];
  for (const prop of fallback) {
    if (features[0].properties && prop in features[0].properties) {
      return prop;
    }
  }
  return preferred;
}

function updateMapLegend(min, max, colorStart, colorEnd) {
  if (!mapLegend) {
    return;
  }
  const minLabel = Math.round(min).toLocaleString("en-US");
  const maxLabel = Math.round(max).toLocaleString("en-US");
  mapLegend.innerHTML = `
    <div class="legend-bar" style="background: linear-gradient(90deg, ${colorStart}, ${colorEnd});"></div>
    <div class="legend-labels">
      <span>${minLabel} ha</span>
      <span>${maxLabel} ha</span>
    </div>
  `;
}

function showMapTooltip(event, text) {
  if (!mapTooltip) {
    return;
  }
  mapTooltip.textContent = text;
  mapTooltip.style.opacity = "1";
  const offset = 12;
  mapTooltip.style.left = `${event.clientX + offset}px`;
  mapTooltip.style.top = `${event.clientY + offset}px`;
}

function hideMapTooltip() {
  if (!mapTooltip) {
    return;
  }
  mapTooltip.style.opacity = "0";
}

async function renderChoropleth(level, valuesByName) {
  if (!mapContainer) {
    return;
  }
  if (!window.d3) {
    setMapPlaceholder("D3 no esta disponible.");
    return;
  }

  const geo = await loadGeoData(level);
  if (!geo || !geo.features) {
    setMapPlaceholder("No se encontro el GeoJSON para el mapa.");
    return;
  }

  ensureMapSvg();
  const { width, height } = mapSize();
  mapSvg.attr("width", width).attr("height", height);

  const nameProp = resolveNameProperty(
    geo.features,
    MAP_SOURCES[level].nameProp,
  );
  const projection = window.d3.geoMercator().fitSize([width, height], geo);
  mapPath = window.d3.geoPath().projection(projection);

  const values = Object.values(valuesByName).filter((v) => Number.isFinite(v));
  const min = values.length ? Math.min(...values) : 0;
  const max = values.length ? Math.max(...values) : 1;
  const scale = window.d3
    .scaleSequential(window.d3.interpolateYlOrRd)
    .domain([min, max || 1]);

  const colorStart = scale(min);
  const colorEnd = scale(max || 1);
  updateMapLegend(min, max || 1, colorStart, colorEnd);

  const features = mapG
    .selectAll("path")
    .data(geo.features, (d) => normalizeKey(d.properties?.[nameProp]));

  features
    .join(
      (enter) =>
        enter
          .append("path")
          .attr("d", mapPath)
          .attr("class", "map-shape")
          .attr("fill", "#1f2430")
          .attr("stroke", "#1b1f2a")
          .attr("stroke-width", 0.6),
      (update) => update.attr("d", mapPath),
      (exit) => exit.remove(),
    )
    .attr("fill", (d) => {
      const key = normalizeKey(d.properties?.[nameProp]);
      const value = valuesByName[key];
      if (!Number.isFinite(value)) {
        return "#1f2430";
      }
      return scale(value);
    })
    .on("mousemove", (event, d) => {
      const key = normalizeKey(d.properties?.[nameProp]);
      const rawName = d.properties?.[nameProp] || "NA";
      const value = valuesByName[key];
      const valueText = Number.isFinite(value)
        ? `${Math.round(value).toLocaleString("en-US")} ha`
        : "Sin datos";
      showMapTooltip(event, `${rawName}: ${valueText}`);
    })
    .on("mouseleave", hideMapTooltip);

  setMapPlaceholder("");
  mapLastLevel = level;
  mapLastValues = valuesByName;
}

function flattenAggregateResults(level, results) {
  if (level === "province") {
    const out = {};
    Object.values(results || {}).forEach((provMap) => {
      Object.entries(provMap || {}).forEach(([prov, val]) => {
        const key = normalizeKey(prov);
        const current = out[key] || 0;
        out[key] = current + Number(val?.pred_ha || 0);
      });
    });
    return out;
  }
  const out = {};
  Object.entries(results || {}).forEach(([name, val]) => {
    out[normalizeKey(name)] = Number(val?.pred_ha || 0);
  });
  return out;
}

async function fetchAggregate(deltas, groupBy) {
  const { mode, year } = getScenario();
  const group = groupBy === "province" ? "province" : "department";
  const body = {
    group_by: group,
    mode,
    overrides: { YEAR: year },
  };
  if (Object.keys(deltas).length > 0) {
    body.feature_deltas = deltas;
  }

  const res = await fetch(`${API_URL}/predict/aggregate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });

  if (!res.ok) {
    throw new Error(`API error (${res.status})`);
  }
  return res.json();
}

async function fetchMarginal(level, deltas) {
  const { mode, year } = getScenario();
  const body = {
    mode,
    overrides: { YEAR: year },
    deltas,
  };

  const res = await fetch(`${API_URL}/marginal/${level}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });

  if (!res.ok) {
    throw new Error(`API error (${res.status})`);
  }
  return res.json();
}

function renderStats(current, bau) {
  totalEl.innerText = current.toLocaleString("en-US", {
    minimumFractionDigits: 0,
    maximumFractionDigits: 0,
  });
  bauEl.innerText = bau.toLocaleString("en-US", {
    minimumFractionDigits: 0,
    maximumFractionDigits: 0,
  });

  const delta = current - bau;
  const sign = delta > 0 ? "+" : "";
  deltaEl.innerText = `${sign}${delta.toLocaleString("en-US", {
    maximumFractionDigits: 0,
  })} ha vs BAU`;

  deltaEl.className = "delta-badge";
  if (delta > 50) {
    deltaEl.classList.add("delta-pos"); // Red
  } else if (delta < -50) {
    deltaEl.classList.add("delta-neg"); // Green
  } else {
    deltaEl.classList.add("delta-neutral");
  }
}

function initCharts() {
  const ctxM = document.getElementById("marginalChart").getContext("2d");
  contribChart = new Chart(ctxM, {
    type: "bar",
    data: {
      labels: [],
      datasets: [
        {
          label: "Contribucion (%)",
          data: [],
          backgroundColor: [],
          meta: [],
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        y: {
          beginAtZero: true,
          max: 100,
          grid: { color: "#30363d" },
          ticks: {
            callback: (value) => `${value}%`,
          },
        },
        x: { grid: { display: false } },
      },
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: (context) => {
              const item = context.dataset.meta?.[context.dataIndex];
              if (!item) {
                return `${context.formattedValue}%`;
              }
              const percent = `${item.percent.toFixed(1)}%`;
              const deltaHa = formatSignedNumber(item.deltaHa, 0);
              const deltaUnit = formatWithUnit(
                item.delta,
                item.unit,
                item.decimals,
              );
              const perUnit = Number.isFinite(item.deltaPerUnit)
                ? formatSignedNumber(item.deltaPerUnit, 2)
                : "n/a";
              return [
                `${item.label}: ${percent}`,
                `Delta ha: ${deltaHa}`,
                `Delta input: ${deltaUnit}`,
                `Delta ha / unit: ${perUnit}`,
              ];
            },
          },
        },
      },
    },
  });

  const ctxD = document.getElementById("deptChart").getContext("2d");
  deptChart = new Chart(ctxD, {
    type: "bar",
    data: {
      labels: [],
      datasets: [
        {
          label: "Deforestacion (ha)",
          data: [],
          backgroundColor: COLOR_ACCENT,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      indexAxis: "y", // Horizontal bar
      scales: {
        x: { grid: { color: "#30363d" } },
        y: { grid: { display: false } },
      },
      plugins: { legend: { display: false } },
    },
  });
}

function flattenResultsForChart(level, results) {
  if (level === "province") {
    const out = {};
    Object.values(results || {}).forEach((provMap) => {
      Object.entries(provMap || {}).forEach(([prov, val]) => {
        out[prov] = (out[prov] || 0) + Number(val?.pred_ha || 0);
      });
    });
    return Object.entries(out).map(([name, value]) => ({ name, value }));
  }
  return Object.entries(results || {}).map(([name, val]) => ({
    name,
    value: Number(val?.pred_ha || 0),
  }));
}

function renderRegionChart(results, level) {
  const items = flattenResultsForChart(level, results)
    .sort((a, b) => b.value - a.value)
    .slice(0, 10);

  deptChart.data.labels = items.map((i) => i.name);
  deptChart.data.datasets[0].data = items.map((i) => i.value);
  deptChart.update();

  if (regionDescEl) {
    regionDescEl.textContent =
      level === "province"
        ? "Deforestacion proyectada por Provincia"
        : "Deforestacion proyectada por Departamento";
  }
}

function updateGroupLabel() {
  if (!groupLabelEl) {
    return;
  }
  groupLabelEl.textContent =
    levelSelect?.value === "province" ? "Provincia" : "Departamento";
}

function buildContributionItems(effects) {
  const items = Object.entries(effects).map(([feature, effect]) => {
    const meta = FEATURE_LABELS[feature] || {
      label: feature,
      unit: "",
      decimals: 0,
    };
    return {
      feature,
      label: meta.label,
      unit: meta.unit,
      decimals: meta.decimals,
      delta: Number(effect?.delta || 0),
      deltaHa: Number(effect?.delta_ha || 0),
      deltaPerUnit: effect?.delta_per_unit,
    };
  });

  const totalAbs = items.reduce((sum, item) => sum + Math.abs(item.deltaHa), 0);

  return items
    .map((item) => ({
      ...item,
      percent: totalAbs > 0 ? (Math.abs(item.deltaHa) / totalAbs) * 100 : 0,
      color: item.deltaHa >= 0 ? "#da3633" : "#238636",
    }))
    .sort((a, b) => b.percent - a.percent);
}

function renderContributionChart(groupName) {
  if (!marginalResults || !marginalResults[groupName]) {
    contribEmptyEl.style.display = "block";
    contribEmptyEl.textContent = "No hay datos para esta seleccion.";
    contribChart.data.labels = [];
    contribChart.data.datasets[0].data = [];
    contribChart.update();
    return;
  }

  const effects = marginalResults[groupName].effects || {};
  const items = buildContributionItems(effects);

  if (!items.length) {
    contribEmptyEl.style.display = "block";
    contribEmptyEl.textContent = "No hay contribuciones para mostrar.";
    contribChart.data.labels = [];
    contribChart.data.datasets[0].data = [];
    contribChart.update();
    return;
  }

  contribEmptyEl.style.display = "none";
  contribChart.data.labels = items.map((item) => item.label);
  contribChart.data.datasets[0].data = items.map((item) => item.percent);
  contribChart.data.datasets[0].backgroundColor = items.map(
    (item) => item.color,
  );
  contribChart.data.datasets[0].meta = items;
  contribChart.update();

  if (contribTitleEl && contribSubtitleEl) {
    const levelLabel =
      levelSelect?.value === "province" ? "Provincia" : "Departamento";
    contribTitleEl.textContent = "Contribuciones por causa";
    contribSubtitleEl.textContent = `${levelLabel}: ${groupName}`;
  }
}

async function updateContributions(deltas) {
  const hasDeltas = Object.keys(deltas).length > 0;
  updateGroupLabel();

  if (!hasDeltas) {
    marginalResults = null;
    contribEmptyEl.style.display = "block";
    contribEmptyEl.textContent =
      "Ajuste los sliders para calcular contribuciones.";
    contribChart.data.labels = [];
    contribChart.data.datasets[0].data = [];
    contribChart.update();
    return;
  }

  try {
    const data = await fetchMarginal(levelSelect.value, deltas);
    marginalResults = data.results || {};
    renderContributionChart(groupSelect.value);
  } catch (err) {
    console.error("Marginal effects failed", err);
    contribEmptyEl.style.display = "block";
    contribEmptyEl.textContent =
      "Error al calcular contribuciones (revise el API).";
  }
}

async function refreshScenario() {
  syncScenarioMeta();
  setSlidersEnabled(false);

  try {
    const bauData = await fetchAggregate({}, getGroupBy());
    bauTotal = bauData.total_pred_ha;
    renderStats(bauTotal, bauTotal);
    renderRegionChart(bauData.results, getGroupBy());
    setSlidersEnabled(true);
    await updateDashboard();
  } catch (err) {
    console.error("Init failed:", err);
    totalEl.innerText = "Error";
    setSlidersEnabled(false);
  }
}

async function updateDashboard() {
  syncScenarioMeta();
  const deltas = buildFeatureDeltas();

  try {
    const data = await fetchAggregate(deltas, getGroupBy());
    currentTotal = data.total_pred_ha;

    renderStats(currentTotal, bauTotal);
    renderRegionChart(data.results, getGroupBy());
    const mapValues = flattenAggregateResults(getGroupBy(), data.results);
    try {
      await renderChoropleth(getGroupBy(), mapValues);
    } catch (err) {
      console.error("Map render failed", err);
      setMapPlaceholder("No se pudo cargar el mapa.");
    }
  } catch (err) {
    console.error("Update failed", err);
  }

  await updateContributions(deltas);
}

// UI Listeners
sliders.forEach((slider) => {
  slider.addEventListener("input", () => {
    updateSliderLabel(slider);
    clearTimeout(debounceTimer);
    debounceTimer = setTimeout(updateDashboard, 500);
  });
});

resetBtn.addEventListener("click", () => {
  sliders.forEach((s) => {
    s.value = 0;
    updateSliderLabel(s);
  });
  updateDashboard();
});

modeSelect.addEventListener("change", refreshScenario);
yearInput.addEventListener("change", refreshScenario);
levelSelect.addEventListener("change", async () => {
  await loadGroupOptions();
  await updateDashboard();
});
groupSelect.addEventListener("change", () => {
  renderContributionChart(groupSelect.value);
});

function closeSidebar() {
  document.body.classList.remove("sidebar-open");
}

if (sidebarToggle) {
  sidebarToggle.addEventListener("click", () => {
    document.body.classList.toggle("sidebar-open");
  });
}
if (sidebarOverlay) {
  sidebarOverlay.addEventListener("click", closeSidebar);
}
window.addEventListener("keydown", (event) => {
  if (event.key === "Escape") {
    closeSidebar();
  }
});
window.addEventListener("resize", () => {
  if (mapLastLevel && mapLastValues) {
    renderChoropleth(mapLastLevel, mapLastValues).catch((err) => {
      console.error("Map resize failed", err);
    });
  }
});

async function init() {
  initCharts();
  initSliderUI();
  updateGroupLabel();
  await loadGroupOptions();
  await refreshScenario();
}

// Start
init();
