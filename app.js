(() => {
  'use strict';

  const ids = [
    'sampleBtn', 'csvFile', 'csvText', 'latColumn', 'lonColumn', 'valueColumn', 'trendBasis',
    'roiFile', 'roiText', 'parseBtn', 'parseRoiBtn', 'deriveRoiBtn', 'variogramModel',
    'lagCount', 'nugget', 'partialSill', 'range', 'neighbors', 'gridSize', 'roiBuffer',
    'fitVariogramBtn', 'runBtn', 'exportCsvBtn', 'exportGeojsonBtn', 'metricPoints',
    'metricCells', 'metricR2', 'metricRmse', 'statusLine', 'mapCanvas', 'legend',
    'variogramCanvas', 'variogramSummary', 'cellSummary', 'readLat', 'readLon',
    'readValue', 'readVariance'
  ];
  const els = {};
  ids.forEach((id) => {
    els[id] = document.getElementById(id);
  });

  const state = {
    headers: [],
    points: [],
    roiGeojson: null,
    roiPolygons: [],
    bbox: null,
    projection: null,
    trend: null,
    variogramBins: [],
    grid: null,
    mapTransform: null
  };

  const colorStops = [
    { t: 0, color: '#244a9b' },
    { t: 0.42, color: '#2f9c95' },
    { t: 0.72, color: '#e7cf4f' },
    { t: 1, color: '#c44931' }
  ];

  const sampleCsv = [
    'id,lat,lon,As_mgkg',
    'S01,10.6680,106.5650,18.4',
    'S02,10.6940,106.5940,21.7',
    'S03,10.7210,106.5480,23.1',
    'S04,10.7440,106.6080,28.8',
    'S05,10.7700,106.5750,31.2',
    'S06,10.8010,106.6200,34.6',
    'S07,10.8300,106.5850,33.8',
    'S08,10.8580,106.6500,37.5',
    'S09,10.6760,106.6810,19.8',
    'S10,10.7070,106.7110,24.2',
    'S11,10.7420,106.7240,35.6',
    'S12,10.7740,106.7040,44.9',
    'S13,10.8050,106.7350,48.1',
    'S14,10.8380,106.7090,45.3',
    'S15,10.8620,106.7600,42.7',
    'S16,10.6500,106.7420,20.3',
    'S17,10.6880,106.7870,26.4',
    'S18,10.7230,106.8050,32.9',
    'S19,10.7590,106.7840,40.6',
    'S20,10.7920,106.8230,43.4',
    'S21,10.8240,106.7950,41.0',
    'S22,10.8500,106.8400,38.2',
    'S23,10.7110,106.6500,30.7',
    'S24,10.7800,106.6550,39.5',
    'S25,10.8180,106.6680,41.9',
    'S26,10.7440,106.8550,35.8',
    'S27,10.7920,106.8750,39.2',
    'S28,10.8580,106.8950,36.5'
  ].join('\n');

  const sampleRoi = {
    type: 'FeatureCollection',
    features: [
      {
        type: 'Feature',
        properties: { name: 'ROI mẫu' },
        geometry: {
          type: 'Polygon',
          coordinates: [[
            [106.515, 10.635],
            [106.650, 10.610],
            [106.815, 10.635],
            [106.935, 10.720],
            [106.920, 10.900],
            [106.750, 10.930],
            [106.585, 10.880],
            [106.510, 10.760],
            [106.515, 10.635]
          ]]
        }
      }
    ]
  };

  function init() {
    setupTabs();
    setupEvents();
    loadSample();
    window.addEventListener('resize', () => {
      drawMap();
      drawVariogram();
    });
  }

  function setupTabs() {
    document.querySelectorAll('.tab-button').forEach((button) => {
      button.addEventListener('click', () => {
        document.querySelectorAll('.tab-button').forEach((b) => b.classList.remove('active'));
        document.querySelectorAll('.tab-panel').forEach((panel) => panel.classList.remove('active'));
        button.classList.add('active');
        document.getElementById('tab-' + button.dataset.tab).classList.add('active');
      });
    });
  }

  function setupEvents() {
    els.sampleBtn.addEventListener('click', loadSample);
    els.csvFile.addEventListener('change', async () => loadFileIntoTextarea(els.csvFile, els.csvText, true));
    els.roiFile.addEventListener('change', async () => loadFileIntoTextarea(els.roiFile, els.roiText, false));
    els.csvText.addEventListener('input', () => updateColumnsFromText());
    els.parseBtn.addEventListener('click', handleParsePoints);
    els.parseRoiBtn.addEventListener('click', handleParseRoi);
    els.deriveRoiBtn.addEventListener('click', handleDeriveRoi);
    els.fitVariogramBtn.addEventListener('click', handleFitVariogram);
    els.runBtn.addEventListener('click', handleRunInterpolation);
    els.exportCsvBtn.addEventListener('click', exportCsv);
    els.exportGeojsonBtn.addEventListener('click', exportGeojson);
    els.mapCanvas.addEventListener('mousemove', handleMapHover);
    els.mapCanvas.addEventListener('mouseleave', clearReadout);
  }

  async function loadFileIntoTextarea(input, textarea, isCsv) {
    const file = input.files && input.files[0];
    if (!file) return;
    const text = await file.text();
    textarea.value = text;
    if (isCsv) {
      updateColumnsFromText();
    }
    setStatus('Đã nạp file ' + file.name + '.', 'ok');
  }

  function loadSample() {
    els.csvText.value = sampleCsv;
    els.roiText.value = JSON.stringify(sampleRoi, null, 2);
    updateColumnsFromText();
    try {
      loadPointsFromUi();
      loadRoiFromUi();
      fitTrendModel();
      computeEmpiricalVariogram();
      fitVariogramParameters();
      drawVariogram();
      runInterpolation(false).then(() => setStatus('Đã nạp và nội suy dữ liệu mẫu.', 'ok'));
    } catch (error) {
      setStatus(error.message, 'warn');
      drawMap();
    }
  }

  function handleParsePoints() {
    try {
      loadPointsFromUi();
      fitTrendModel();
      computeEmpiricalVariogram();
      drawMap();
      drawVariogram();
      updateMetrics();
      setStatus('Đã cập nhật ' + state.points.length + ' điểm đo.', 'ok');
    } catch (error) {
      setStatus(error.message, 'warn');
    }
  }

  function handleParseRoi() {
    try {
      loadRoiFromUi();
      drawMap();
      setStatus('Đã cập nhật ROI.', 'ok');
    } catch (error) {
      setStatus(error.message, 'warn');
    }
  }

  function handleDeriveRoi() {
    try {
      if (!state.points.length) loadPointsFromUi();
      const roi = makeBufferedRoi(pointBBox(state.points), readNumber(els.roiBuffer, 12) / 100);
      els.roiText.value = JSON.stringify(roi, null, 2);
      loadRoiFromUi();
      drawMap();
      setStatus('Đã tạo ROI từ vùng bao điểm đo.', 'ok');
    } catch (error) {
      setStatus(error.message, 'warn');
    }
  }

  function handleFitVariogram() {
    try {
      loadPointsFromUi();
      ensureRoi();
      fitTrendModel();
      computeEmpiricalVariogram();
      fitVariogramParameters();
      drawVariogram();
      drawMap();
      setStatus('Đã fit variogram cho phần dư hồi quy.', 'ok');
    } catch (error) {
      setStatus(error.message, 'warn');
    }
  }

  async function handleRunInterpolation() {
    try {
      loadPointsFromUi();
      ensureRoi();
      fitTrendModel();
      computeEmpiricalVariogram();
      drawVariogram();
      await runInterpolation(true);
      setStatus('Hoàn tất nội suy Regression Kriging.', 'ok');
    } catch (error) {
      setStatus(error.message, 'warn');
    }
  }

  function updateColumnsFromText() {
    try {
      const parsed = parseDelimited(els.csvText.value);
      state.headers = parsed.headers;
      updateColumnOptions(parsed.headers, true);
    } catch (_) {
      state.headers = [];
    }
  }

  function loadPointsFromUi() {
    const parsed = parseDelimited(els.csvText.value);
    if (parsed.headers.length < 3) throw new Error('CSV cần tối thiểu ba cột: lat, lon và chỉ tiêu.');
    state.headers = parsed.headers;
    updateColumnOptions(parsed.headers, true);

    const latKey = els.latColumn.value;
    const lonKey = els.lonColumn.value;
    const valueKey = els.valueColumn.value;
    if (!latKey || !lonKey || !valueKey) throw new Error('Chưa chọn đủ cột lat, lon và chỉ tiêu.');

    const merged = new Map();
    parsed.records.forEach((record) => {
      const lat = toNumber(record[latKey]);
      const lon = toNumber(record[lonKey]);
      const value = toNumber(record[valueKey]);
      if (!Number.isFinite(lat) || !Number.isFinite(lon) || !Number.isFinite(value)) return;
      if (lat < -90 || lat > 90 || lon < -180 || lon > 180) return;
      const key = lat.toFixed(8) + '|' + lon.toFixed(8);
      if (!merged.has(key)) {
        merged.set(key, { lat, lon, valueSum: 0, count: 0, value: 0 });
      }
      const item = merged.get(key);
      item.valueSum += value;
      item.count += 1;
      item.value = item.valueSum / item.count;
    });

    const points = Array.from(merged.values()).map((p, index) => ({
      id: 'P' + String(index + 1).padStart(2, '0'),
      lat: p.lat,
      lon: p.lon,
      value: p.value,
      x: 0,
      y: 0,
      trend: 0,
      residual: 0
    }));

    if (points.length < 4) throw new Error('Cần ít nhất 4 điểm hợp lệ để kriging ổn định.');
    state.points = points;
    if (!state.roiPolygons.length) {
      state.bbox = pointBBox(points);
    }
    updateMetrics();
    return points;
  }

  function loadRoiFromUi() {
    if (!els.roiText.value.trim()) throw new Error('ROI đang trống. Có thể dùng nút Từ điểm đo.');
    const geojson = JSON.parse(els.roiText.value);
    const polygons = extractPolygons(geojson);
    if (!polygons.length) throw new Error('ROI phải là GeoJSON Polygon hoặc MultiPolygon.');
    state.roiGeojson = geojson;
    state.roiPolygons = polygons;
    state.bbox = polygonBBox(polygons);
    return polygons;
  }

  function ensureRoi() {
    if (els.roiText.value.trim()) {
      loadRoiFromUi();
      return;
    }
    const roi = makeBufferedRoi(pointBBox(state.points), readNumber(els.roiBuffer, 12) / 100);
    els.roiText.value = JSON.stringify(roi, null, 2);
    loadRoiFromUi();
  }

  function parseDelimited(text) {
    const cleaned = text.replace(/^\uFEFF/, '').trim();
    if (!cleaned) return { headers: [], records: [] };
    const lines = cleaned.split(/\r?\n/).filter((line) => line.trim().length);
    const delimiter = detectDelimiter(lines[0]);
    const rows = lines.map((line) => parseDelimitedLine(line, delimiter));
    const headers = rows[0].map((header, index) => {
      const name = header.trim();
      return name || 'col_' + (index + 1);
    });
    const records = rows.slice(1).map((row) => {
      const record = {};
      headers.forEach((header, index) => {
        record[header] = row[index] === undefined ? '' : row[index];
      });
      return record;
    });
    return { headers, records };
  }

  function detectDelimiter(line) {
    const candidates = [',', ';', '\t'];
    let best = ',';
    let bestCount = -1;
    candidates.forEach((candidate) => {
      let count = 0;
      let inQuotes = false;
      for (let i = 0; i < line.length; i += 1) {
        const ch = line[i];
        if (ch === '"') inQuotes = !inQuotes;
        if (ch === candidate && !inQuotes) count += 1;
      }
      if (count > bestCount) {
        best = candidate;
        bestCount = count;
      }
    });
    return best;
  }

  function parseDelimitedLine(line, delimiter) {
    const cells = [];
    let current = '';
    let inQuotes = false;
    for (let i = 0; i < line.length; i += 1) {
      const ch = line[i];
      if (ch === '"') {
        if (inQuotes && line[i + 1] === '"') {
          current += '"';
          i += 1;
        } else {
          inQuotes = !inQuotes;
        }
      } else if (ch === delimiter && !inQuotes) {
        cells.push(current.trim());
        current = '';
      } else {
        current += ch;
      }
    }
    cells.push(current.trim());
    return cells;
  }

  function updateColumnOptions(headers, keepCurrent) {
    const current = {
      lat: els.latColumn.value,
      lon: els.lonColumn.value,
      value: els.valueColumn.value
    };
    fillSelect(els.latColumn, headers);
    fillSelect(els.lonColumn, headers);
    fillSelect(els.valueColumn, headers);
    const guesses = guessColumns(headers);
    els.latColumn.value = keepCurrent && headers.includes(current.lat) ? current.lat : guesses.lat;
    els.lonColumn.value = keepCurrent && headers.includes(current.lon) ? current.lon : guesses.lon;
    els.valueColumn.value = keepCurrent && headers.includes(current.value) ? current.value : guesses.value;
  }

  function fillSelect(select, headers) {
    select.innerHTML = '';
    headers.forEach((header) => {
      const option = document.createElement('option');
      option.value = header;
      option.textContent = header;
      select.appendChild(option);
    });
  }

  function guessColumns(headers) {
    const normalized = headers.map((header) => normalizeName(header));
    const latIndex = findHeader(normalized, ['lat', 'latitude', 'vido', 'vi_do', 'y']);
    const lonIndex = findHeader(normalized, ['lon', 'lng', 'long', 'longitude', 'kinhdo', 'kinh_do', 'x']);
    let valueIndex = headers.length - 1;
    for (let i = 0; i < headers.length; i += 1) {
      if (i !== latIndex && i !== lonIndex) valueIndex = i;
    }
    return {
      lat: headers[Math.max(0, latIndex)],
      lon: headers[Math.max(0, lonIndex)],
      value: headers[Math.max(0, valueIndex)]
    };
  }

  function findHeader(normalized, keys) {
    for (let i = 0; i < normalized.length; i += 1) {
      if (keys.some((key) => normalized[i] === key || normalized[i].includes(key))) return i;
    }
    return 0;
  }

  function normalizeName(value) {
    return String(value).toLowerCase().normalize('NFD').replace(/[\u0300-\u036f]/g, '').replace(/[^a-z0-9_]/g, '');
  }

  function toNumber(value) {
    if (typeof value === 'number') return value;
    let text = String(value ?? '').trim().replace(/\s/g, '');
    if (!text) return NaN;
    if (/^-?\d+,\d+$/.test(text)) text = text.replace(',', '.');
    if (/^-?\d{1,3}(,\d{3})+(\.\d+)?$/.test(text)) text = text.replace(/,/g, '');
    return Number(text);
  }

  function extractPolygons(geojson) {
    const polygons = [];
    function addGeometry(geometry) {
      if (!geometry) return;
      if (geometry.type === 'Polygon') {
        addPolygon(geometry.coordinates);
      } else if (geometry.type === 'MultiPolygon') {
        geometry.coordinates.forEach(addPolygon);
      } else if (geometry.type === 'GeometryCollection') {
        geometry.geometries.forEach(addGeometry);
      }
    }
    function addPolygon(rawPolygon) {
      const polygon = rawPolygon.map((ring) => ring.map((coord) => [Number(coord[0]), Number(coord[1])]).filter((coord) => Number.isFinite(coord[0]) && Number.isFinite(coord[1]))).filter((ring) => ring.length >= 4);
      if (polygon.length) polygons.push(polygon);
    }
    if (geojson.type === 'FeatureCollection') {
      geojson.features.forEach((feature) => addGeometry(feature.geometry));
    } else if (geojson.type === 'Feature') {
      addGeometry(geojson.geometry);
    } else {
      addGeometry(geojson);
    }
    return polygons;
  }

  function pointInPolygons(lon, lat, polygons) {
    return polygons.some((polygon) => pointInPolygon(lon, lat, polygon));
  }

  function pointInPolygon(lon, lat, polygon) {
    if (!pointInRing(lon, lat, polygon[0])) return false;
    for (let i = 1; i < polygon.length; i += 1) {
      if (pointInRing(lon, lat, polygon[i])) return false;
    }
    return true;
  }

  function pointInRing(lon, lat, ring) {
    let inside = false;
    for (let i = 0, j = ring.length - 1; i < ring.length; j = i, i += 1) {
      const xi = ring[i][0];
      const yi = ring[i][1];
      const xj = ring[j][0];
      const yj = ring[j][1];
      const intersect = yi > lat !== yj > lat && lon < ((xj - xi) * (lat - yi)) / (yj - yi + 1e-15) + xi;
      if (intersect) inside = !inside;
    }
    return inside;
  }

  function pointBBox(points) {
    if (!points.length) throw new Error('Chưa có điểm đo hợp lệ.');
    return points.reduce((box, p) => ({
      minLon: Math.min(box.minLon, p.lon),
      maxLon: Math.max(box.maxLon, p.lon),
      minLat: Math.min(box.minLat, p.lat),
      maxLat: Math.max(box.maxLat, p.lat)
    }), { minLon: Infinity, maxLon: -Infinity, minLat: Infinity, maxLat: -Infinity });
  }

  function polygonBBox(polygons) {
    const box = { minLon: Infinity, maxLon: -Infinity, minLat: Infinity, maxLat: -Infinity };
    polygons.forEach((polygon) => {
      polygon.forEach((ring) => {
        ring.forEach((coord) => {
          box.minLon = Math.min(box.minLon, coord[0]);
          box.maxLon = Math.max(box.maxLon, coord[0]);
          box.minLat = Math.min(box.minLat, coord[1]);
          box.maxLat = Math.max(box.maxLat, coord[1]);
        });
      });
    });
    return box;
  }

  function makeBufferedRoi(box, bufferRatio) {
    const lonRange = Math.max(box.maxLon - box.minLon, 0.01);
    const latRange = Math.max(box.maxLat - box.minLat, 0.01);
    const lonPad = lonRange * bufferRatio;
    const latPad = latRange * bufferRatio;
    const minLon = box.minLon - lonPad;
    const maxLon = box.maxLon + lonPad;
    const minLat = box.minLat - latPad;
    const maxLat = box.maxLat + latPad;
    return {
      type: 'FeatureCollection',
      features: [{
        type: 'Feature',
        properties: { name: 'ROI từ điểm đo' },
        geometry: {
          type: 'Polygon',
          coordinates: [[[minLon, minLat], [maxLon, minLat], [maxLon, maxLat], [minLon, maxLat], [minLon, minLat]]]
        }
      }]
    };
  }

  function makeProjection(box) {
    const lat0 = (box.minLat + box.maxLat) / 2;
    const lon0 = (box.minLon + box.maxLon) / 2;
    const cosLat = Math.max(0.05, Math.cos((lat0 * Math.PI) / 180));
    return { lat0, lon0, mPerLat: 111320, mPerLon: 111320 * cosLat };
  }

  function project(lon, lat, projection) {
    return {
      x: (lon - projection.lon0) * projection.mPerLon,
      y: (lat - projection.lat0) * projection.mPerLat
    };
  }

  function fitTrendModel() {
    const box = state.bbox || pointBBox(state.points);
    state.projection = makeProjection(box);
    state.points.forEach((point) => {
      const projected = project(point.lon, point.lat, state.projection);
      point.x = projected.x;
      point.y = projected.y;
    });

    const basis = els.trendBasis.value;
    const rows = state.points.map((point) => trendFeatures(point.x, point.y, basis));
    const values = state.points.map((point) => point.value);
    const beta = solveNormalEquation(rows, values);
    const mean = values.reduce((sum, value) => sum + value, 0) / values.length;
    const trend = { basis, beta, mean, r2: 0, rmse: 0, residualVariance: 0 };

    let ssResidual = 0;
    let ssTotal = 0;
    state.points.forEach((point) => {
      const estimate = beta ? dot(trendFeatures(point.x, point.y, basis), beta) : mean;
      point.trend = estimate;
      point.residual = point.value - estimate;
      ssResidual += point.residual * point.residual;
      ssTotal += (point.value - mean) * (point.value - mean);
    });
    trend.r2 = ssTotal > 0 ? 1 - ssResidual / ssTotal : 1;
    trend.rmse = Math.sqrt(ssResidual / state.points.length);
    const residualMean = state.points.reduce((sum, point) => sum + point.residual, 0) / state.points.length;
    trend.residualVariance = Math.max(1e-9, state.points.reduce((sum, point) => sum + Math.pow(point.residual - residualMean, 2), 0) / Math.max(1, state.points.length - 1));
    state.trend = trend;
    updateMetrics();
    return trend;
  }

  function trendFeatures(x, y, basis) {
    const xk = x / 1000;
    const yk = y / 1000;
    if (basis === 'quadratic') {
      return [1, xk, yk, xk * xk, xk * yk, yk * yk];
    }
    return [1, xk, yk];
  }

  function solveNormalEquation(rows, values) {
    if (!rows.length) return null;
    const m = rows[0].length;
    const matrix = Array.from({ length: m }, () => Array(m).fill(0));
    const rhs = Array(m).fill(0);
    rows.forEach((row, rowIndex) => {
      for (let i = 0; i < m; i += 1) {
        rhs[i] += row[i] * values[rowIndex];
        for (let j = 0; j < m; j += 1) matrix[i][j] += row[i] * row[j];
      }
    });
    const trace = matrix.reduce((sum, row, index) => sum + Math.abs(row[index]), 0);
    const ridge = trace * 1e-9 + 1e-9;
    for (let i = 0; i < m; i += 1) matrix[i][i] += ridge;
    return solveLinearSystem(matrix, rhs);
  }

  function computeEmpiricalVariogram() {
    if (!state.trend) fitTrendModel();
    const lagCount = clamp(Math.round(readNumber(els.lagCount, 12)), 5, 24);
    const pairs = [];
    let maxDistance = 0;
    for (let i = 0; i < state.points.length; i += 1) {
      for (let j = i + 1; j < state.points.length; j += 1) {
        const h = distance(state.points[i], state.points[j]);
        if (h <= 0) continue;
        const gamma = 0.5 * Math.pow(state.points[i].residual - state.points[j].residual, 2);
        pairs.push({ h, gamma });
        maxDistance = Math.max(maxDistance, h);
      }
    }
    if (!pairs.length) {
      state.variogramBins = [];
      return [];
    }
    const width = maxDistance / lagCount;
    const bins = Array.from({ length: lagCount }, () => ({ sumH: 0, sumG: 0, count: 0 }));
    pairs.forEach((pair) => {
      const index = Math.min(lagCount - 1, Math.floor(pair.h / width));
      bins[index].sumH += pair.h;
      bins[index].sumG += pair.gamma;
      bins[index].count += 1;
    });
    state.variogramBins = bins.filter((bin) => bin.count > 0).map((bin) => ({
      h: bin.sumH / bin.count,
      gamma: bin.sumG / bin.count,
      count: bin.count
    }));
    return state.variogramBins;
  }

  function fitVariogramParameters() {
    const bins = state.variogramBins.length ? state.variogramBins : computeEmpiricalVariogram();
    const model = els.variogramModel.value;
    const variance = state.trend ? state.trend.residualVariance : 1;
    if (!bins.length) {
      els.nugget.value = formatInput(variance * 0.05);
      els.partialSill.value = formatInput(variance * 0.95);
      els.range.value = '1000';
      return;
    }
    const maxH = Math.max(...bins.map((bin) => bin.h));
    const rangeFactors = [0.18, 0.25, 0.33, 0.45, 0.6, 0.8, 1.0, 1.25, 1.55];
    const nuggetFractions = [0, 0.03, 0.08, 0.15, 0.25, 0.4];
    let best = null;
    rangeFactors.forEach((factor) => {
      const range = Math.max(maxH * factor, 1);
      nuggetFractions.forEach((fraction) => {
        const nugget = Math.max(0, variance * fraction);
        let numerator = 0;
        let denominator = 0;
        bins.forEach((bin) => {
          const f = modelShape(bin.h, range, model);
          const weight = Math.max(1, bin.count);
          numerator += weight * f * (bin.gamma - nugget);
          denominator += weight * f * f;
        });
        const partialSill = Math.max(1e-6, numerator / Math.max(denominator, 1e-12));
        let score = 0;
        bins.forEach((bin) => {
          const estimate = nugget + partialSill * modelShape(bin.h, range, model);
          score += Math.max(1, bin.count) * Math.pow(bin.gamma - estimate, 2);
        });
        if (!best || score < best.score) best = { nugget, partialSill, range, score };
      });
    });
    els.nugget.value = formatInput(best.nugget);
    els.partialSill.value = formatInput(best.partialSill);
    els.range.value = String(Math.round(best.range));
  }

  async function runInterpolation(showProgress) {
    const params = readVariogramParams();
    const box = state.bbox || pointBBox(state.points);
    const projection = state.projection || makeProjection(box);
    const widthM = Math.max(1, (box.maxLon - box.minLon) * projection.mPerLon);
    const heightM = Math.max(1, (box.maxLat - box.minLat) * projection.mPerLat);
    const cols = clamp(Math.round(readNumber(els.gridSize, 80)), 25, 180);
    const rows = clamp(Math.round(cols * heightM / widthM), 18, 180);
    const stepLon = (box.maxLon - box.minLon) / cols;
    const stepLat = (box.maxLat - box.minLat) / rows;
    const cells = [];
    const cellMap = new Array(rows * cols).fill(null);
    const totalRows = rows;

    for (let row = 0; row < rows; row += 1) {
      const lat = box.maxLat - (row + 0.5) * stepLat;
      for (let col = 0; col < cols; col += 1) {
        const lon = box.minLon + (col + 0.5) * stepLon;
        if (!pointInPolygons(lon, lat, state.roiPolygons)) continue;
        const prediction = predictAt(lon, lat, params);
        const cell = { row, col, lat, lon, ...prediction };
        cells.push(cell);
        cellMap[row * cols + col] = cell;
      }
      if (showProgress && row % 8 === 0) {
        setStatus('Đang nội suy hàng ' + (row + 1) + '/' + totalRows + '...', 'ok');
        await nextFrame();
      }
    }

    if (!cells.length) throw new Error('Không có ô lưới nào nằm trong ROI. Kiểm tra lại ROI và tọa độ điểm.');
    state.grid = { bbox: box, rows, cols, stepLon, stepLat, cells, cellMap, params };
    drawMap();
    drawVariogram();
    updateMetrics();
  }

  function predictAt(lon, lat, params) {
    const projected = project(lon, lat, state.projection);
    const trendValue = state.trend.beta ? dot(trendFeatures(projected.x, projected.y, state.trend.basis), state.trend.beta) : state.trend.mean;
    const kriged = krigeResidual(projected.x, projected.y, params);
    return {
      value: trendValue + kriged.residual,
      trend: trendValue,
      residual: kriged.residual,
      variance: kriged.variance
    };
  }

  function krigeResidual(x, y, params) {
    const neighbors = clamp(Math.round(readNumber(els.neighbors, 18)), 4, 64);
    const nearby = state.points.map((point) => ({ point, d: Math.hypot(point.x - x, point.y - y) })).sort((a, b) => a.d - b.d).slice(0, Math.min(neighbors, state.points.length));
    if (nearby[0] && nearby[0].d < 1e-7) return { residual: nearby[0].point.residual, variance: 0 };
    const n = nearby.length;
    if (n < 3) return idwResidual(x, y, nearby, params);
    const size = n + 1;
    const matrix = Array.from({ length: size }, () => Array(size).fill(0));
    const rhs = Array(size).fill(0);
    for (let i = 0; i < n; i += 1) {
      for (let j = 0; j < n; j += 1) {
        const d = Math.hypot(nearby[i].point.x - nearby[j].point.x, nearby[i].point.y - nearby[j].point.y);
        matrix[i][j] = covariance(d, params);
      }
      matrix[i][n] = 1;
      matrix[n][i] = 1;
      rhs[i] = covariance(nearby[i].d, params);
    }
    rhs[n] = 1;
    const solution = solveLinearSystem(matrix, rhs);
    if (!solution) return idwResidual(x, y, nearby, params);
    let residual = 0;
    let lambdaC = 0;
    for (let i = 0; i < n; i += 1) {
      residual += solution[i] * nearby[i].point.residual;
      lambdaC += solution[i] * rhs[i];
    }
    const variance = Math.max(0, covariance(0, params) - lambdaC - solution[n]);
    if (!Number.isFinite(residual) || !Number.isFinite(variance)) return idwResidual(x, y, nearby, params);
    return { residual, variance };
  }

  function idwResidual(x, y, nearby, params) {
    let weighted = 0;
    let sumWeights = 0;
    nearby.forEach((item) => {
      const weight = 1 / Math.max(1e-9, Math.pow(Math.hypot(item.point.x - x, item.point.y - y), 2));
      weighted += weight * item.point.residual;
      sumWeights += weight;
    });
    return { residual: weighted / Math.max(sumWeights, 1e-12), variance: covariance(0, params) };
  }

  function readVariogramParams() {
    const nugget = Math.max(0, readNumber(els.nugget, 0));
    const partialSill = Math.max(1e-9, readNumber(els.partialSill, 1));
    const range = Math.max(1, readNumber(els.range, 1000));
    return { model: els.variogramModel.value, nugget, partialSill, range };
  }

  function modelShape(h, range, model) {
    const r = Math.max(0, h) / Math.max(1e-9, range);
    if (model === 'exponential') return 1 - Math.exp(-3 * r);
    if (model === 'gaussian') return 1 - Math.exp(-3 * r * r);
    if (r >= 1) return 1;
    return 1.5 * r - 0.5 * r * r * r;
  }

  function semivariance(h, params) {
    if (h <= 1e-9) return 0;
    return params.nugget + params.partialSill * modelShape(h, params.range, params.model);
  }

  function covariance(h, params) {
    const totalSill = params.nugget + params.partialSill;
    if (h <= 1e-9) return totalSill;
    return Math.max(0, totalSill - semivariance(h, params));
  }

  function solveLinearSystem(matrix, rhs) {
    const n = rhs.length;
    const a = matrix.map((row, i) => row.slice().concat(rhs[i]));
    for (let col = 0; col < n; col += 1) {
      let pivot = col;
      for (let row = col + 1; row < n; row += 1) {
        if (Math.abs(a[row][col]) > Math.abs(a[pivot][col])) pivot = row;
      }
      if (Math.abs(a[pivot][col]) < 1e-12) return null;
      if (pivot !== col) {
        const tmp = a[col];
        a[col] = a[pivot];
        a[pivot] = tmp;
      }
      const pivotValue = a[col][col];
      for (let c = col; c <= n; c += 1) a[col][c] /= pivotValue;
      for (let row = 0; row < n; row += 1) {
        if (row === col) continue;
        const factor = a[row][col];
        if (Math.abs(factor) < 1e-16) continue;
        for (let c = col; c <= n; c += 1) a[row][c] -= factor * a[col][c];
      }
    }
    return a.map((row) => row[n]);
  }

  function drawMap() {
    const canvas = els.mapCanvas;
    const setup = prepareCanvas(canvas);
    const ctx = setup.ctx;
    const w = setup.w;
    const h = setup.h;
    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = '#f8fbfa';
    ctx.fillRect(0, 0, w, h);

    const box = (state.grid && state.grid.bbox) || state.bbox || (state.points.length ? pointBBox(state.points) : null);
    if (!box) {
      drawCenteredText(ctx, w, h, 'Chưa có dữ liệu');
      updateLegend(null, null);
      return;
    }

    const transform = makeMapTransform(box, w, h);
    state.mapTransform = transform;
    drawSubtleGrid(ctx, transform, w, h);

    if (state.grid) drawGridCells(ctx, transform);
    if (state.roiPolygons.length) drawRoi(ctx, transform);
    if (state.points.length) drawSamplePoints(ctx, transform);
    const values = state.grid ? state.grid.cells.map((cell) => cell.value) : state.points.map((point) => point.value);
    updateLegend(Math.min(...values), Math.max(...values));
  }

  function prepareCanvas(canvas) {
    const rect = canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    const width = Math.max(320, rect.width || canvas.width);
    const height = Math.max(220, rect.height || canvas.height);
    if (canvas.width !== Math.round(width * dpr) || canvas.height !== Math.round(height * dpr)) {
      canvas.width = Math.round(width * dpr);
      canvas.height = Math.round(height * dpr);
    }
    const ctx = canvas.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return { ctx, w: width, h: height };
  }

  function makeMapTransform(box, w, h) {
    const projection = makeProjection(box);
    const padding = Math.max(22, Math.min(w, h) * 0.055);
    const widthM = Math.max(1, (box.maxLon - box.minLon) * projection.mPerLon);
    const heightM = Math.max(1, (box.maxLat - box.minLat) * projection.mPerLat);
    const scale = Math.min((w - padding * 2) / widthM, (h - padding * 2) / heightM);
    return {
      box,
      projection,
      scale,
      x0: (w - widthM * scale) / 2,
      y0: (h - heightM * scale) / 2,
      widthM,
      heightM
    };
  }

  function coordToPixel(lon, lat, transform) {
    return {
      x: transform.x0 + (lon - transform.box.minLon) * transform.projection.mPerLon * transform.scale,
      y: transform.y0 + (transform.box.maxLat - lat) * transform.projection.mPerLat * transform.scale
    };
  }

  function drawSubtleGrid(ctx, transform, w, h) {
    ctx.save();
    ctx.strokeStyle = '#e5ece8';
    ctx.lineWidth = 1;
    const lines = 6;
    for (let i = 0; i <= lines; i += 1) {
      const x = transform.x0 + (transform.widthM * transform.scale * i) / lines;
      const y = transform.y0 + (transform.heightM * transform.scale * i) / lines;
      ctx.beginPath();
      ctx.moveTo(x, transform.y0);
      ctx.lineTo(x, transform.y0 + transform.heightM * transform.scale);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(transform.x0, y);
      ctx.lineTo(transform.x0 + transform.widthM * transform.scale, y);
      ctx.stroke();
    }
    ctx.restore();
  }

  function drawGridCells(ctx, transform) {
    const cells = state.grid.cells;
    const min = Math.min(...cells.map((cell) => cell.value));
    const max = Math.max(...cells.map((cell) => cell.value));
    ctx.save();
    cells.forEach((cell) => {
      const lon0 = state.grid.bbox.minLon + cell.col * state.grid.stepLon;
      const lon1 = lon0 + state.grid.stepLon;
      const lat1 = state.grid.bbox.maxLat - cell.row * state.grid.stepLat;
      const lat0 = lat1 - state.grid.stepLat;
      const p0 = coordToPixel(lon0, lat0, transform);
      const p1 = coordToPixel(lon1, lat1, transform);
      ctx.fillStyle = colorForValue(cell.value, min, max);
      ctx.globalAlpha = 0.86;
      ctx.fillRect(p0.x, p1.y, Math.max(1, p1.x - p0.x + 0.7), Math.max(1, p0.y - p1.y + 0.7));
    });
    ctx.restore();
  }

  function drawRoi(ctx, transform) {
    ctx.save();
    ctx.lineWidth = 2;
    ctx.strokeStyle = '#10231d';
    ctx.fillStyle = 'rgba(255,255,255,0.04)';
    state.roiPolygons.forEach((polygon) => {
      polygon.forEach((ring, ringIndex) => {
        ctx.beginPath();
        ring.forEach((coord, index) => {
          const p = coordToPixel(coord[0], coord[1], transform);
          if (index === 0) ctx.moveTo(p.x, p.y);
          else ctx.lineTo(p.x, p.y);
        });
        ctx.closePath();
        if (ringIndex === 0) ctx.fill();
        ctx.stroke();
      });
    });
    ctx.restore();
  }

  function drawSamplePoints(ctx, transform) {
    const min = Math.min(...state.points.map((point) => point.value));
    const max = Math.max(...state.points.map((point) => point.value));
    ctx.save();
    state.points.forEach((point) => {
      const p = coordToPixel(point.lon, point.lat, transform);
      ctx.beginPath();
      ctx.arc(p.x, p.y, 4.8, 0, Math.PI * 2);
      ctx.fillStyle = colorForValue(point.value, min, max);
      ctx.fill();
      ctx.lineWidth = 1.5;
      ctx.strokeStyle = '#ffffff';
      ctx.stroke();
      ctx.beginPath();
      ctx.arc(p.x, p.y, 5.6, 0, Math.PI * 2);
      ctx.strokeStyle = 'rgba(13, 28, 23, 0.55)';
      ctx.stroke();
    });
    ctx.restore();
  }

  function drawVariogram() {
    const canvas = els.variogramCanvas;
    const setup = prepareCanvas(canvas);
    const ctx = setup.ctx;
    const w = setup.w;
    const h = setup.h;
    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(0, 0, w, h);
    const bins = state.variogramBins;
    if (!bins.length) {
      drawCenteredText(ctx, w, h, 'Chưa có variogram');
      els.variogramSummary.textContent = 'Chưa tính';
      return;
    }
    const params = readVariogramParams();
    const pad = { left: 46, right: 14, top: 16, bottom: 34 };
    const maxX = Math.max(params.range * 1.15, ...bins.map((bin) => bin.h)) * 1.05;
    const maxY = Math.max(params.nugget + params.partialSill, ...bins.map((bin) => bin.gamma)) * 1.2;
    const xScale = (w - pad.left - pad.right) / maxX;
    const yScale = (h - pad.top - pad.bottom) / maxY;
    const px = (x) => pad.left + x * xScale;
    const py = (y) => h - pad.bottom - y * yScale;

    ctx.save();
    ctx.strokeStyle = '#d9e2dd';
    ctx.lineWidth = 1;
    ctx.fillStyle = '#63716b';
    ctx.font = '12px sans-serif';
    for (let i = 0; i <= 4; i += 1) {
      const y = (maxY * i) / 4;
      ctx.beginPath();
      ctx.moveTo(pad.left, py(y));
      ctx.lineTo(w - pad.right, py(y));
      ctx.stroke();
      ctx.fillText(formatShort(y), 6, py(y) + 4);
    }
    for (let i = 0; i <= 4; i += 1) {
      const x = (maxX * i) / 4;
      ctx.beginPath();
      ctx.moveTo(px(x), pad.top);
      ctx.lineTo(px(x), h - pad.bottom);
      ctx.stroke();
      ctx.fillText(formatShort(x / 1000) + ' km', px(x) - 16, h - 10);
    }

    ctx.strokeStyle = '#15211d';
    ctx.lineWidth = 1.3;
    ctx.beginPath();
    ctx.moveTo(pad.left, pad.top);
    ctx.lineTo(pad.left, h - pad.bottom);
    ctx.lineTo(w - pad.right, h - pad.bottom);
    ctx.stroke();

    ctx.strokeStyle = '#1f7a5c';
    ctx.lineWidth = 2.2;
    ctx.beginPath();
    for (let i = 0; i <= 100; i += 1) {
      const x = (maxX * i) / 100;
      const y = semivariance(x, params);
      if (i === 0) ctx.moveTo(px(x), py(y));
      else ctx.lineTo(px(x), py(y));
    }
    ctx.stroke();

    bins.forEach((bin) => {
      const radius = clamp(3 + Math.sqrt(bin.count) * 0.35, 3.5, 8);
      ctx.beginPath();
      ctx.arc(px(bin.h), py(bin.gamma), radius, 0, Math.PI * 2);
      ctx.fillStyle = '#c44931';
      ctx.fill();
      ctx.strokeStyle = '#ffffff';
      ctx.lineWidth = 1.2;
      ctx.stroke();
    });
    ctx.restore();
    els.variogramSummary.textContent = params.model + ' | nugget ' + formatShort(params.nugget) + ' | sill ' + formatShort(params.nugget + params.partialSill) + ' | range ' + formatShort(params.range / 1000) + ' km';
  }

  function handleMapHover(event) {
    if (!state.grid || !state.mapTransform) return;
    const rect = els.mapCanvas.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;
    const tr = state.mapTransform;
    const lon = tr.box.minLon + (x - tr.x0) / (tr.scale * tr.projection.mPerLon);
    const lat = tr.box.maxLat - (y - tr.y0) / (tr.scale * tr.projection.mPerLat);
    const col = Math.floor((lon - state.grid.bbox.minLon) / state.grid.stepLon);
    const row = Math.floor((state.grid.bbox.maxLat - lat) / state.grid.stepLat);
    if (row < 0 || col < 0 || row >= state.grid.rows || col >= state.grid.cols) {
      clearReadout();
      return;
    }
    const cell = state.grid.cellMap[row * state.grid.cols + col];
    if (!cell) {
      clearReadout();
      return;
    }
    els.cellSummary.textContent = 'Hàng ' + (row + 1) + ', cột ' + (col + 1);
    els.readLat.textContent = cell.lat.toFixed(6);
    els.readLon.textContent = cell.lon.toFixed(6);
    els.readValue.textContent = formatShort(cell.value);
    els.readVariance.textContent = formatShort(cell.variance);
  }

  function clearReadout() {
    els.cellSummary.textContent = '-';
    els.readLat.textContent = '-';
    els.readLon.textContent = '-';
    els.readValue.textContent = '-';
    els.readVariance.textContent = '-';
  }

  function exportCsv() {
    if (!state.grid || !state.grid.cells.length) {
      setStatus('Chưa có kết quả nội suy để xuất.', 'warn');
      return;
    }
    const rows = ['lon,lat,prediction,kriging_variance,trend,residual'];
    state.grid.cells.forEach((cell) => {
      rows.push([cell.lon, cell.lat, cell.value, cell.variance, cell.trend, cell.residual].map((value) => Number(value).toPrecision(10)).join(','));
    });
    download('regression_kriging_grid.csv', rows.join('\n'), 'text/csv;charset=utf-8');
    setStatus('Đã xuất CSV kết quả.', 'ok');
  }

  function exportGeojson() {
    if (!state.grid || !state.grid.cells.length) {
      setStatus('Chưa có kết quả nội suy để xuất.', 'warn');
      return;
    }
    const geojson = {
      type: 'FeatureCollection',
      features: state.grid.cells.map((cell) => ({
        type: 'Feature',
        properties: {
          prediction: cell.value,
          kriging_variance: cell.variance,
          trend: cell.trend,
          residual: cell.residual,
          row: cell.row,
          col: cell.col
        },
        geometry: { type: 'Point', coordinates: [cell.lon, cell.lat] }
      }))
    };
    download('regression_kriging_grid.geojson', JSON.stringify(geojson, null, 2), 'application/geo+json;charset=utf-8');
    setStatus('Đã xuất GeoJSON kết quả.', 'ok');
  }

  function download(filename, content, mime) {
    const blob = new Blob([content], { type: mime });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    link.remove();
    URL.revokeObjectURL(url);
  }

  function updateLegend(min, max) {
    if (!Number.isFinite(min) || !Number.isFinite(max)) {
      els.legend.innerHTML = '<strong>Bản đồ</strong><div>Chưa có kết quả</div>';
      return;
    }
    els.legend.innerHTML = '<strong>Giá trị nội suy</strong><div class="legend-bar"></div><div class="legend-scale"><span>' + formatShort(min) + '</span><span>' + formatShort(max) + '</span></div>';
  }

  function updateMetrics() {
    els.metricPoints.textContent = String(state.points.length);
    els.metricCells.textContent = state.grid ? String(state.grid.cells.length) : '0';
    els.metricR2.textContent = state.trend ? formatShort(state.trend.r2) : '-';
    els.metricRmse.textContent = state.trend ? formatShort(state.trend.rmse) : '-';
  }

  function setStatus(message, type) {
    els.statusLine.textContent = message;
    els.statusLine.classList.remove('ok', 'warn');
    if (type) els.statusLine.classList.add(type);
  }

  function readNumber(input, fallback) {
    const value = Number(input.value);
    return Number.isFinite(value) ? value : fallback;
  }

  function distance(a, b) {
    return Math.hypot(a.x - b.x, a.y - b.y);
  }

  function dot(a, b) {
    let sum = 0;
    for (let i = 0; i < a.length; i += 1) sum += a[i] * b[i];
    return sum;
  }

  function clamp(value, min, max) {
    return Math.max(min, Math.min(max, value));
  }

  function nextFrame() {
    return new Promise((resolve) => requestAnimationFrame(resolve));
  }

  function formatInput(value) {
    if (!Number.isFinite(value)) return '0';
    if (Math.abs(value) >= 100) return value.toFixed(0);
    if (Math.abs(value) >= 10) return value.toFixed(2);
    if (Math.abs(value) >= 1) return value.toFixed(3);
    return value.toPrecision(3);
  }

  function formatShort(value) {
    if (!Number.isFinite(value)) return '-';
    const abs = Math.abs(value);
    if (abs >= 1000) return value.toLocaleString('vi-VN', { maximumFractionDigits: 0 });
    if (abs >= 100) return value.toLocaleString('vi-VN', { maximumFractionDigits: 1 });
    if (abs >= 10) return value.toLocaleString('vi-VN', { maximumFractionDigits: 2 });
    if (abs >= 1) return value.toLocaleString('vi-VN', { maximumFractionDigits: 3 });
    return value.toLocaleString('vi-VN', { maximumSignificantDigits: 3 });
  }

  function colorForValue(value, min, max) {
    const t = max > min ? clamp((value - min) / (max - min), 0, 1) : 0.5;
    for (let i = 1; i < colorStops.length; i += 1) {
      const left = colorStops[i - 1];
      const right = colorStops[i];
      if (t <= right.t) {
        const local = (t - left.t) / Math.max(1e-12, right.t - left.t);
        return mixColor(left.color, right.color, local);
      }
    }
    return colorStops[colorStops.length - 1].color;
  }

  function mixColor(a, b, t) {
    const ca = hexToRgb(a);
    const cb = hexToRgb(b);
    const rgb = ca.map((value, index) => Math.round(value + (cb[index] - value) * t));
    return 'rgb(' + rgb.join(',') + ')';
  }

  function hexToRgb(hex) {
    const clean = hex.replace('#', '');
    return [0, 2, 4].map((index) => parseInt(clean.slice(index, index + 2), 16));
  }

  function drawCenteredText(ctx, w, h, text) {
    ctx.save();
    ctx.fillStyle = '#63716b';
    ctx.font = '600 15px sans-serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(text, w / 2, h / 2);
    ctx.restore();
  }

  init();
})();
