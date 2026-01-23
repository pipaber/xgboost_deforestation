const API_URL = "http://localhost:8000";
const DEFAULT_YEAR = 2024;

const SLIDER_CONFIG = {
  Mining: {
    feature: "Miner\u00eda",
    unit: "ha",
    min: -2000,
    max: 2000,
    step: 1,
    decimals: 0,
  },
  Agriculture: {
    feature: "area_agropec",
    unit: "ha",
    min: -20000,
    max: 20000,
    step: 1,
    decimals: 0,
  },
  Infrastructure: {
    feature: "Infraestructura",
    unit: "ha",
    min: -500,
    max: 500,
    step: 1,
    decimals: 0,
  },
  Temperature: {
    feature: "tmean",
    unit: "\u00b0C",
    min: -2,
    max: 2,
    step: 0.1,
    decimals: 1,
  },
  Precipitation: {
    feature: "pp",
    unit: "mm",
    min: -500,
    max: 500,
    step: 10,
    decimals: 0,
  },
  Socioeconomic: {
    feature: "Poblaci\u00f3n",
    unit: "personas",
    min: -20000,
    max: 20000,
    step: 500,
    decimals: 0,
  },
};

const FEATURE_LABELS = {
  "Miner\u00eda": { label: "Mineria", unit: "ha", decimals: 0 },
  area_agropec: { label: "Agricultura", unit: "ha", decimals: 0 },
  Infraestructura: { label: "Infraestructura", unit: "ha", decimals: 0 },
  tmean: { label: "Temperatura", unit: "\u00b0C", decimals: 1 },
  pp: { label: "Precipitacion", unit: "mm", decimals: 0 },
  "Poblaci\u00f3n": { label: "Poblacion", unit: "personas", decimals: 0 },
  Poblacion: { label: "Poblacion", unit: "personas", decimals: 0 },
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

const MAP_RAMP = [
  "#F4D166",
  "#F1D065",
  "#EDCF64",
  "#EACF64",
  "#E7CE63",
  "#E4CD62",
  "#E1CC62",
  "#DECC61",
  "#DACB60",
  "#D7CA60",
  "#D4C95F",
  "#D1C95F",
  "#CEC85E",
  "#CBC75E",
  "#C8C65E",
  "#C5C55D",
  "#C2C55D",
  "#BFC45C",
  "#BBC45C",
  "#B8C35C",
  "#B5C25B",
  "#B2C25B",
  "#AFC15B",
  "#ACC05B",
  "#A9BF5A",
  "#A6BE5A",
  "#A3BD5A",
  "#A0BC5A",
  "#9DBC59",
  "#9ABB59",
  "#97BA58",
  "#94B958",
  "#91B858",
  "#8EB758",
  "#8BB657",
  "#88B557",
  "#85B457",
  "#83B357",
  "#80B357",
  "#7DB257",
  "#7BB156",
  "#78B056",
  "#75AF56",
  "#72AD56",
  "#6FAC56",
  "#6DAB56",
  "#6AAA56",
  "#67A956",
  "#64A856",
  "#62A756",
  "#5FA555",
  "#5DA455",
  "#5AA355",
  "#58A255",
  "#56A154",
  "#54A054",
  "#529F54",
  "#509E53",
  "#4E9D53",
  "#4C9C52",
  "#4B9A52",
  "#499952",
  "#489851",
  "#469651",
  "#459550",
  "#449450",
  "#429350",
  "#41924F",
  "#40914F",
  "#3F8F4F",
  "#3D8E4E",
  "#3C8D4E",
  "#3B8B4D",
  "#3A8A4D",
  "#39894C",
  "#38884C",
  "#36864B",
  "#35854B",
  "#34844A",
  "#338349",
  "#328248",
  "#308047",
  "#2F7F46",
  "#2E7D45",
  "#2C7C44",
  "#2B7B43",
  "#297A42",
  "#277942",
  "#267841",
  "#247740",
  "#23753F",
  "#21743E",
  "#1F733D",
  "#1D723C",
  "#1C713B",
  "#1A703A",
  "#196F39",
  "#176E38",
  "#166D37",
  "#146C36",
];

const MAP_EMPTY_FILL = "#e6edd6";

const controlInputs = document.querySelectorAll(".control-input");
const stepButtons = document.querySelectorAll(".step-btn");
const resetBtn = document.getElementById("reset-btn");
const modeSelect = document.getElementById("mode-select");
const yearInput = document.getElementById("year-input");
const scenarioMetaEl = document.getElementById("scenario-meta");
const scenarioPillEl = document.getElementById("scenario-pill");
const totalEl = document.getElementById("total-ha");
const deltaEl = document.getElementById("total-delta");
const provinceSubtitleEl = document.getElementById("province-subtitle");
const selectedDeptEl = document.getElementById("selected-dept");
const impactBubblesEl = document.getElementById("impact-bubbles");
const impactNoteEl = document.getElementById("impact-note");
const impactSubtitleEl = document.getElementById("impact-subtitle");
const mapTooltip = document.getElementById("map-tooltip");
const sidebarToggle = document.getElementById("sidebar-toggle");
const sidebarOverlay = document.getElementById("sidebar-overlay");

const deptMapState = createMapState({
  container: document.getElementById("dept-map"),
  legend: document.getElementById("dept-legend"),
  placeholder: document.getElementById("dept-placeholder"),
});
const provMapState = createMapState({
  container: document.getElementById("prov-map"),
  legend: document.getElementById("prov-legend"),
  placeholder: document.getElementById("prov-placeholder"),
});

let bauTotal = 0;
let currentTotal = 0;
let debounceTimer = null;
let selectedDepartment = null;
let selectedDepartmentKey = null;
let lastDepartmentTotals = {};
let lastProvinceResults = null;
let lastMarginalResults = null;
let lastDeltas = {};
const mapGeoCache = {};

function createMapState({ container, legend, placeholder }) {
  return {
    container,
    legend,
    placeholder,
    svg: null,
    g: null,
    path: null,
    lastRender: null,
  };
}

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

function normalizeKey(value) {
  return String(value || "")
    .normalize("NFD")
    .replace(/[\u0300-\u036f]/g, "")
    .trim()
    .toUpperCase();
}

function setControlsEnabled(enabled) {
  controlInputs.forEach((input) => {
    input.disabled = !enabled;
    input.closest(".control-group")?.classList.toggle("is-disabled", !enabled);
  });
  stepButtons.forEach((button) => {
    button.disabled = !enabled;
  });
  if (resetBtn) {
    resetBtn.disabled = !enabled;
  }
}

function initControlUI() {
  controlInputs.forEach((input) => {
    const config = SLIDER_CONFIG[input.dataset.driver];
    if (!config) {
      input.disabled = true;
      input.closest(".control-group")?.classList.add("is-disabled");
      return;
    }
    input.min = config.min;
    input.max = config.max;
    input.step = config.step;
    input.value = 0;
    updateControlValue(input, true);
  });
}

function clampValue(value, min, max) {
  if (!Number.isFinite(value)) {
    return 0;
  }
  return Math.min(max, Math.max(min, value));
}

function parseInputValue(input) {
  const raw = String(input.value || "").trim();
  if (raw === "" || raw === "-" || raw === "+") {
    return null;
  }
  const value = Number(raw);
  return Number.isFinite(value) ? value : null;
}

function normalizeInputValue(input, config, commit = false) {
  const parsed = parseInputValue(input);
  if (parsed === null) {
    return null;
  }
  const clamped = clampValue(parsed, config.min, config.max);
  const normalized = Math.abs(clamped) < 1e-9 ? 0 : clamped;
  if (commit) {
    input.value = String(normalized);
  }
  return normalized;
}

function updateControlValue(input, commit = false) {
  const config = SLIDER_CONFIG[input.dataset.driver];
  if (!config) {
    return;
  }
  const value = normalizeInputValue(input, config, commit);
  if (value === null) {
    return;
  }
  const valueLabel = input
    .closest(".control-group")
    ?.querySelector("[data-value]");
  if (!valueLabel) {
    return;
  }
  valueLabel.textContent = formatWithUnit(value, config.unit, config.decimals);
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

function syncScenarioMeta() {
  const { mode, year } = getScenario();
  if (scenarioMetaEl) {
    scenarioMetaEl.textContent = `Modo: ${mode} \u00b7 Ano: ${year}`;
  }
  if (scenarioPillEl) {
    scenarioPillEl.textContent = `${mode} \u00b7 ${year}`;
  }
}

function buildFeatureDeltas() {
  const deltas = {};
  controlInputs.forEach((input) => {
    const config = SLIDER_CONFIG[input.dataset.driver];
    if (!config) {
      return;
    }
    const value = normalizeInputValue(input, config, false);
    if (value === null) {
      return;
    }
    if (!Number.isFinite(value) || value === 0) {
      return;
    }
    deltas[config.feature] = value;
  });
  return deltas;
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

function mapSize(container) {
  const rect = container.getBoundingClientRect();
  const width = Math.max(280, Math.floor(rect.width));
  const height = Math.max(240, Math.floor(rect.height));
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

function setMapPlaceholder(state, message) {
  if (!state.placeholder) {
    return;
  }
  state.placeholder.textContent = message;
  state.placeholder.style.display = message ? "flex" : "none";
  if (message && state.legend) {
    state.legend.innerHTML = "";
  }
}

function ensureMapSvg(state) {
  if (!state.container) {
    return;
  }
  if (!state.svg) {
    state.svg = window.d3
      .select(state.container)
      .append("svg")
      .attr("class", "choropleth");
    state.g = state.svg.append("g");
  }
}

function updateMapLegend(legendEl, min, max, colorStart, colorEnd) {
  if (!legendEl) {
    return;
  }
  const minLabel = Math.round(min).toLocaleString("en-US");
  const maxLabel = Math.round(max).toLocaleString("en-US");
  legendEl.innerHTML = `
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

function renderChoropleth(state, geo, nameProp, valuesByName, options = {}) {
  if (!state.container) {
    return;
  }
  if (!window.d3) {
    setMapPlaceholder(state, "D3 no esta disponible.");
    return;
  }
  if (!geo || !geo.features) {
    setMapPlaceholder(state, "No se encontro el GeoJSON.");
    return;
  }

  const features = options.filter
    ? geo.features.filter(options.filter)
    : geo.features.slice();
  if (!features.length) {
    setMapPlaceholder(
      state,
      options.emptyMessage || "No hay datos para mostrar.",
    );
    return;
  }

  ensureMapSvg(state);
  const { width, height } = mapSize(state.container);
  state.svg.attr("width", width).attr("height", height);

  const fitTarget = options.fitGeo || { type: "FeatureCollection", features };
  const projection = window.d3
    .geoMercator()
    .fitSize([width, height], fitTarget);
  const path = window.d3.geoPath().projection(projection);
  state.path = path;

  const values = Object.values(valuesByName || {}).filter((v) =>
    Number.isFinite(v),
  );
  const min = values.length ? Math.min(...values) : 0;
  const max = values.length ? Math.max(...values) : 1;
  const scale = window.d3
    .scaleSequential(window.d3.interpolateRgbBasis(MAP_RAMP))
    .domain([min, max || 1]);

  updateMapLegend(state.legend, min, max || 1, scale(min), scale(max || 1));

  const joinKey = (d) => normalizeKey(d.properties?.[nameProp]);
  const selection = state.g.selectAll("path.map-shape").data(features, joinKey);
  const merged = selection.join(
    (enter) => enter.append("path").attr("class", "map-shape"),
    (update) => update,
    (exit) => exit.remove(),
  );

  merged
    .attr("d", path)
    .attr("fill", (d) => {
      const key = normalizeKey(d.properties?.[nameProp]);
      const value = valuesByName?.[key];
      if (!Number.isFinite(value)) {
        return MAP_EMPTY_FILL;
      }
      return scale(value);
    })
    .classed(
      "is-selected",
      (d) => normalizeKey(d.properties?.[nameProp]) === options.selectedKey,
    )
    .on("mousemove", (event, d) => {
      const key = normalizeKey(d.properties?.[nameProp]);
      const rawName = d.properties?.[nameProp] || "NA";
      const value = valuesByName?.[key];
      const valueText = Number.isFinite(value)
        ? `${Math.round(value).toLocaleString("en-US")} ha`
        : "Sin datos";
      showMapTooltip(event, `${rawName}: ${valueText}`);
    })
    .on("mouseleave", hideMapTooltip);

  if (options.onClick) {
    merged.on("click", (event, d) => options.onClick(d));
  } else {
    merged.on("click", null);
  }

  const outlineData = options.outlineGeo ? [options.outlineGeo] : [];
  state.g
    .selectAll("path.map-outline")
    .data(outlineData)
    .join(
      (enter) => enter.append("path").attr("class", "map-outline"),
      (update) => update,
      (exit) => exit.remove(),
    )
    .attr("d", path);

  setMapPlaceholder(state, "");
  state.lastRender = { geo, nameProp, valuesByName, options };
}

function computeDepartmentTotals(results) {
  const totals = {};
  Object.entries(results || {}).forEach(([dep, provMap]) => {
    const sum = Object.values(provMap || {}).reduce(
      (acc, item) => acc + Number(item?.pred_ha || 0),
      0,
    );
    totals[normalizeKey(dep)] = sum;
  });
  return totals;
}

function findProvinceMap(results, targetKey) {
  if (!results || !targetKey) {
    return null;
  }
  for (const [dep, provMap] of Object.entries(results)) {
    if (normalizeKey(dep) === targetKey) {
      return { dep, provMap };
    }
  }
  return null;
}

function pickDefaultDepartment(results) {
  let bestKey = null;
  let bestValue = -Infinity;
  Object.entries(results || {}).forEach(([dep, provMap]) => {
    const sum = Object.values(provMap || {}).reduce(
      (acc, item) => acc + Number(item?.pred_ha || 0),
      0,
    );
    if (sum > bestValue) {
      bestValue = sum;
      bestKey = dep;
    }
  });
  return bestKey;
}

async function renderDepartmentMap(valuesByName) {
  const geo = await loadGeoData("department");
  if (!geo?.features) {
    setMapPlaceholder(deptMapState, "No se encontro el mapa de departamentos.");
    return;
  }
  const nameProp = resolveNameProperty(
    geo.features,
    MAP_SOURCES.department.nameProp,
  );

  renderChoropleth(deptMapState, geo, nameProp, valuesByName, {
    selectedKey: selectedDepartmentKey,
    onClick: (feature) => {
      const rawName = feature?.properties?.[nameProp];
      if (rawName) {
        setSelectedDepartment(rawName);
      }
    },
  });
}

async function renderProvinceMap() {
  if (!selectedDepartmentKey) {
    setMapPlaceholder(
      provMapState,
      "Seleccione un departamento para cargar el detalle.",
    );
    if (provMapState.legend) {
      provMapState.legend.innerHTML = "";
    }
    return;
  }

  const geo = await loadGeoData("province");
  if (!geo?.features) {
    setMapPlaceholder(provMapState, "No se encontro el mapa provincial.");
    return;
  }
  const nameProp = resolveNameProperty(
    geo.features,
    MAP_SOURCES.province.nameProp,
  );

  const match = findProvinceMap(lastProvinceResults, selectedDepartmentKey);
  const provinceValues = {};
  const provinceKeys = new Set();
  if (match) {
    Object.entries(match.provMap || {}).forEach(([prov, item]) => {
      const key = normalizeKey(prov);
      provinceValues[key] = Number(item?.pred_ha || 0);
      provinceKeys.add(key);
    });
  }

  if (!provinceKeys.size) {
    setMapPlaceholder(
      provMapState,
      "No hay provincias para este departamento.",
    );
    if (provMapState.legend) {
      provMapState.legend.innerHTML = "";
    }
    return;
  }

  const deptGeo = await loadGeoData("department");
  let outlineFeature = null;
  if (deptGeo?.features?.length) {
    const depNameProp = resolveNameProperty(
      deptGeo.features,
      MAP_SOURCES.department.nameProp,
    );
    outlineFeature =
      deptGeo.features.find(
        (feature) =>
          normalizeKey(feature?.properties?.[depNameProp]) ===
          selectedDepartmentKey,
      ) || null;
  }

  renderChoropleth(provMapState, geo, nameProp, provinceValues, {
    filter: (feature) =>
      provinceKeys.has(normalizeKey(feature?.properties?.[nameProp])),
    fitGeo: outlineFeature || {
      type: "FeatureCollection",
      features: geo.features.filter((feature) =>
        provinceKeys.has(normalizeKey(feature?.properties?.[nameProp])),
      ),
    },
    outlineGeo: outlineFeature,
  });
}

function renderStats(current, bau) {
  if (totalEl) {
    totalEl.textContent = Math.round(current).toLocaleString("en-US");
  }
  if (deltaEl) {
    const delta = current - bau;
    deltaEl.textContent = `${formatSignedNumber(delta, 0)} ha vs BAU`;
    deltaEl.classList.toggle("is-negative", delta < 0);
  }
}

function setSelectedDepartment(name) {
  selectedDepartment = name;
  selectedDepartmentKey = normalizeKey(name);
  if (selectedDeptEl) {
    selectedDeptEl.textContent = name;
  }
  if (provinceSubtitleEl) {
    provinceSubtitleEl.textContent = `Provincias en ${name}`;
  }
  renderDepartmentMap(lastDepartmentTotals).catch((err) => {
    console.error("Department map update failed", err);
  });
  renderProvinceMap().catch((err) => {
    console.error("Province map update failed", err);
  });
  renderImpactBubbles(lastMarginalResults, lastDeltas);
}

function buildImpactItems(effects, deltas) {
  return Object.keys(SLIDER_CONFIG).map((driver) => {
    const config = SLIDER_CONFIG[driver];
    const feature = config.feature;
    const meta = FEATURE_LABELS[feature] || {
      label: feature,
      unit: "",
      decimals: 0,
    };
    const effect = effects?.[feature] || {};
    return {
      label: meta.label,
      feature,
      unit: meta.unit,
      decimals: meta.decimals,
      deltaInput: Number(deltas?.[feature] || 0),
      deltaHa: Number(effect?.delta_ha || 0),
    };
  });
}

function mixColor(startHex, endHex, weight) {
  const clamp = (val) => Math.max(0, Math.min(1, val));
  const w = clamp(weight);
  const parseHex = (hex) => {
    const clean = hex.replace("#", "");
    const num = Number.parseInt(clean, 16);
    return {
      r: (num >> 16) & 255,
      g: (num >> 8) & 255,
      b: num & 255,
    };
  };
  const start = parseHex(startHex);
  const end = parseHex(endHex);
  const r = Math.round(start.r + (end.r - start.r) * w);
  const g = Math.round(start.g + (end.g - start.g) * w);
  const b = Math.round(start.b + (end.b - start.b) * w);
  return `rgb(${r}, ${g}, ${b})`;
}

function renderImpactBubbles(results, deltas) {
  if (!impactBubblesEl) {
    return;
  }

  const effectsForDept = (() => {
    if (!results || !selectedDepartmentKey) {
      return null;
    }
    for (const [dep, payload] of Object.entries(results)) {
      if (normalizeKey(dep) === selectedDepartmentKey) {
        return payload?.effects || null;
      }
    }
    return null;
  })();

  const items = buildImpactItems(effectsForDept, deltas || {});
  const activeItems = items.filter((item) => item.deltaInput !== 0);
  const totalAbs = items.reduce((sum, item) => sum + Math.abs(item.deltaHa), 0);
  const maxAbs = items.reduce(
    (max, item) => Math.max(max, Math.abs(item.deltaHa)),
    0,
  );

  impactBubblesEl.innerHTML = "";
  items.forEach((item) => {
    const intensity = maxAbs > 0 ? Math.abs(item.deltaHa) / maxAbs : 0;
    const isActive = item.deltaInput !== 0;
    const size = isActive ? 46 + intensity * 28 : 36;
    const percent =
      totalAbs > 0 ? (Math.abs(item.deltaHa) / totalAbs) * 100 : 0;

    const color = isActive
      ? item.deltaHa >= 0
        ? mixColor("#f1d065", "#6fac56", intensity)
        : mixColor("#a6be5a", "#2f7f46", intensity)
      : "#e3ebd4";

    const bubble = document.createElement("div");
    bubble.className = "bubble";

    const circle = document.createElement("div");
    circle.className = "bubble-circle";
    circle.style.setProperty("--size", `${size}px`);
    circle.style.setProperty("--color", color);

    const label = document.createElement("div");
    label.className = "bubble-label";
    label.textContent = item.label;

    const meta = document.createElement("div");
    meta.className = "bubble-meta";
    if (isActive) {
      const deltaHa = formatSignedNumber(item.deltaHa, 0);
      const percentText = `${percent.toFixed(0)}%`;
      meta.textContent = `${deltaHa} ha \u00b7 ${percentText}`;
    } else {
      meta.textContent = "Sin cambio";
    }

    bubble.append(circle, label, meta);
    impactBubblesEl.appendChild(bubble);
  });

  if (impactNoteEl) {
    impactNoteEl.style.display = activeItems.length ? "none" : "block";
  }
  if (impactSubtitleEl) {
    impactSubtitleEl.textContent = selectedDepartment
      ? `Impacto relativo en ${selectedDepartment}`
      : "Impacto relativo nacional";
  }
}

async function updateMarginal(deltas) {
  if (!Object.keys(deltas).length) {
    lastMarginalResults = null;
    renderImpactBubbles(lastMarginalResults, deltas);
    return;
  }
  try {
    const data = await fetchMarginal("department", deltas);
    lastMarginalResults = data.results || null;
  } catch (err) {
    console.error("Marginal effects failed", err);
    lastMarginalResults = null;
  }
  renderImpactBubbles(lastMarginalResults, deltas);
}

async function refreshScenario() {
  syncScenarioMeta();
  setControlsEnabled(false);

  try {
    const bauData = await fetchAggregate({}, "province");
    bauTotal = Number(bauData?.total_pred_ha || 0);
    setControlsEnabled(true);
    await updateDashboard();
  } catch (err) {
    console.error("Init failed:", err);
    if (totalEl) {
      totalEl.textContent = "Error";
    }
    setControlsEnabled(false);
  }
}

async function updateDashboard() {
  syncScenarioMeta();
  const deltas = buildFeatureDeltas();
  lastDeltas = deltas;

  try {
    const data = await fetchAggregate(deltas, "province");
    currentTotal = Number(data?.total_pred_ha || 0);
    lastProvinceResults = data?.results || null;

    lastDepartmentTotals = computeDepartmentTotals(lastProvinceResults);
    renderStats(currentTotal, bauTotal);

    const selectedMatch = selectedDepartmentKey
      ? findProvinceMap(lastProvinceResults, selectedDepartmentKey)
      : null;
    if (!selectedDepartment || !selectedMatch) {
      const defaultDept = pickDefaultDepartment(lastProvinceResults);
      if (defaultDept) {
        selectedDepartment = defaultDept;
        selectedDepartmentKey = normalizeKey(defaultDept);
      }
    }

    if (selectedDepartment) {
      if (selectedDeptEl) {
        selectedDeptEl.textContent = selectedDepartment;
      }
      if (provinceSubtitleEl) {
        provinceSubtitleEl.textContent = `Provincias en ${selectedDepartment}`;
      }
    }

    await renderDepartmentMap(lastDepartmentTotals);
    await renderProvinceMap();
  } catch (err) {
    console.error("Update failed", err);
  }

  await updateMarginal(deltas);
}

function scheduleUpdate(delayMs) {
  clearTimeout(debounceTimer);
  debounceTimer = setTimeout(updateDashboard, delayMs);
}

function applyDeltaToInput(input, delta) {
  const config = SLIDER_CONFIG[input.dataset.driver];
  if (!config) {
    return;
  }
  const current = normalizeInputValue(input, config, false) ?? 0;
  const next = clampValue(current + delta, config.min, config.max);
  input.value = String(next);
  updateControlValue(input, true);
  scheduleUpdate(200);
}

controlInputs.forEach((input) => {
  input.addEventListener("input", () => {
    updateControlValue(input, false);
    scheduleUpdate(350);
  });
  input.addEventListener("change", () => {
    updateControlValue(input, true);
    scheduleUpdate(200);
  });
});

stepButtons.forEach((button) => {
  button.addEventListener("click", () => {
    const group = button.closest(".control-group");
    const input = group?.querySelector(".control-input");
    if (!input || input.disabled) {
      return;
    }
    const step = Number(button.dataset.step);
    const delta = Number.isFinite(step) ? step : 0;
    applyDeltaToInput(input, delta);
  });
});

if (resetBtn) {
  resetBtn.addEventListener("click", () => {
    controlInputs.forEach((input) => {
      input.value = 0;
      updateControlValue(input, true);
    });
    updateDashboard();
  });
}

if (modeSelect) {
  modeSelect.addEventListener("change", refreshScenario);
}

if (yearInput) {
  yearInput.addEventListener("change", refreshScenario);
}

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
  const dept = deptMapState.lastRender;
  if (dept) {
    renderChoropleth(
      deptMapState,
      dept.geo,
      dept.nameProp,
      dept.valuesByName,
      dept.options,
    );
  }
  const prov = provMapState.lastRender;
  if (prov) {
    renderChoropleth(
      provMapState,
      prov.geo,
      prov.nameProp,
      prov.valuesByName,
      prov.options,
    );
  }
});

async function init() {
  initControlUI();
  syncScenarioMeta();
  await refreshScenario();
}

init();
