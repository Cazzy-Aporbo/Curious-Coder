const $ = selector => document.querySelector(selector);
const status = text => { $('#status').textContent = text; };
const element = (tag, text) => { const item = document.createElement(tag); if (text !== undefined) item.textContent = String(text); return item; };
const svgElement = (tag, attributes = {}) => {
  const item = document.createElementNS('http://www.w3.org/2000/svg', tag);
  for (const [key, value] of Object.entries(attributes)) item.setAttribute(key, value);
  return item;
};
async function fetchJSON(path) {
  const response = await fetch(path);
  if (!response.ok) throw new Error(`Request failed (${response.status}): ${path}`);
  return response.json();
}
function table(rows, columns, inspectEvent) {
  if (!rows.length) return element('p', 'No records in this fixture.');
  const wrapper = element('div'); wrapper.className = 'table-wrap';
  const result = element('table');
  const heading = element('tr');
  for (const [label] of columns) heading.append(element('th', label));
  const head = element('thead'); head.append(heading); result.append(head);
  const body = element('tbody');
  for (const row of rows) {
    const line = element('tr');
    for (const [, key] of columns) {
      const cell = element('td');
      if (key.endsWith('_event') && inspectEvent && row[key]) {
        const button = element('button', `#${row[key]}`);
        button.addEventListener('click', () => inspectEvent(row[key]).catch(error => status(error.message)));
        cell.append(button);
      } else cell.textContent = String(row[key] ?? 'Not supplied');
      line.append(cell);
    }
    body.append(line);
  }
  result.append(body); wrapper.append(result); return wrapper;
}

async function initialize() {
  let snapshot, live = false, summary, simulation;
  try {
    summary = await fetchJSON('/api/summary');
    simulation = await fetchJSON('/api/simulation');
    live = true;
  } catch {
    snapshot = await fetchJSON(new URL('../studies/results/facility.json', import.meta.url));
    summary = { assets: snapshot.assets, observation_counts: snapshot.counts, audit_events: snapshot.audit_verification.events };
    simulation = { airflow: snapshot.airflow, drift: snapshot.drift };
  }
  $('#mode').textContent = live ? 'Local database inspection API' : 'Recorded synthetic snapshot (not a live database)';
  for (const [label, value] of [['Accepted', summary.observation_counts.accepted], ['Quarantined', summary.observation_counts.quarantined], ['Signed events', summary.audit_events]]) {
    const badge = element('div', `${label}: ${value}`); badge.className = 'badge'; $('#summary').append(badge);
  }
  async function inspectEvent(sequence) {
    const event = live ? await fetchJSON(`/api/events/${sequence}`) : snapshot.events.find(value => value.sequence === sequence);
    $('#event-detail').textContent = JSON.stringify({...event, envelope: JSON.parse(event.envelope)}, null, 2);
    $('#event-detail').parentElement.open = true;
  }
  async function inspectAsset(identifier) {
    const detail = live ? await fetchJSON(`/api/assets/${encodeURIComponent(identifier)}`) : snapshot.asset_details[identifier];
    $('#asset-title').textContent = `${detail.asset.name} · ${identifier}`;
    const target = $('#asset-detail'); target.replaceChildren();
    target.append(element('p', detail.asset.provenance));
    for (const [title, rows, columns] of [
      ['Sensors', detail.sensors, [['Sensor', 'sensor_id'], ['Metric', 'metric'], ['Input unit', 'input_unit']]],
      ['Calibration history', detail.calibrations, [['Record', 'calibration_id'], ['From', 'valid_from'], ['Until (exclusive)', 'valid_until'], ['Uncertainty', 'uncertainty_decimal'], ['Unit', 'uncertainty_unit'], ['Source', 'certificate_reference'], ['Audit event', 'recorded_event']]],
      ['Maintenance', detail.maintenance, [['When', 'performed_at'], ['Activity', 'activity'], ['Audit event', 'recorded_event']]],
      ['Molecular references', detail.sample_links, [['Accession', 'accession'], ['Frame', 'coordinate_frame'], ['Evidence', 'evidence_type']]],
    ]) { target.append(element('h3', title), table(rows, columns, inspectEvent)); }
  }
  for (const asset of summary.assets) {
    const button = element('button', asset.asset_id);
    button.addEventListener('click', () => inspectAsset(asset.asset_id).catch(error => status(error.message)));
    $('#asset-buttons').append(button);
  }
  $('#verify').addEventListener('click', async () => {
    try {
      const result = live ? await fetchJSON('/api/audit/verify') : snapshot.audit_verification;
      status(live ? (result.valid ? `Signatures and trusted head verified; projection match: ${result.projection_matches_signed_digest}.` : `Verification failed: ${result.reason}`)
                  : `Recorded verification at generation time: ${result.valid}. Start the local API to verify the current database.`);
    } catch (error) { status(error.message); }
  });
  const airflow = simulation.airflow;
  const positions = { supply: [35, 175], exhaust: [960, 175] };
  const roomNames = new Set(airflow.rooms.map(room => room.name));
  for (const [index, room] of airflow.rooms.entries()) positions[room.name] = [245 + 460 * index / Math.max(1, airflow.rooms.length - 1), 60 + 230 * index / Math.max(1, airflow.rooms.length - 1)];
  const links = [];
  for (const room of airflow.rooms) {
    links.push({source: 'supply', target: room.name, value: room.supply_m3_h});
    links.push({source: room.name, target: 'exhaust', value: room.exhaust_m3_h});
  }
  for (const flow of airflow.flows) links.push({ source: flow.source, target: flow.target, value: flow.air_m3_h });
  const graph = $('#airflow');
  for (const link of links) {
    const [sx, sy] = positions[link.source], [tx, ty] = positions[link.target];
    const path = svgElement('path', { d: `M ${sx + 120} ${sy + 35} C ${(sx + tx + 120) / 2} ${sy + 35}, ${(sx + tx + 120) / 2} ${ty + 35}, ${tx} ${ty + 35}`,
      fill: 'none', stroke: '#66a99e', 'stroke-width': link.value * .025, opacity: '.48' });
    const title = svgElement('title'); title.textContent = `${link.source} → ${link.target}: ${link.value} m³/h`; path.append(title); graph.append(path);
  }
  const nodeValues = new Map();
  for (const [name, [x, y]] of Object.entries(positions)) {
    const group = svgElement('g', {transform: `translate(${x},${y})`});
    const rect = svgElement('rect', {width: 120, height: 76, rx: 10, fill: '#dceae0', stroke: '#176f68'});
    const label = svgElement('text', {x: 60, y: 23, 'text-anchor': 'middle', class: 'node-label'}); label.textContent = name;
    const value = svgElement('text', {x: 60, y: 45, 'text-anchor': 'middle', class: 'node-value'});
    const concentration = svgElement('text', {x: 60, y: 64, 'text-anchor': 'middle', class: 'node-value'});
    group.append(rect, label, value, concentration);
    if (roomNames.has(name)) {
      group.setAttribute('role', 'button'); group.setAttribute('tabindex', '0'); group.setAttribute('aria-label', `Inspect ${name}`);
      group.addEventListener('click', () => inspectAsset(name).catch(error => status(error.message)));
      group.addEventListener('keydown', event => { if (['Enter', ' '].includes(event.key)) { event.preventDefault(); group.dispatchEvent(new Event('click')); } });
      nodeValues.set(name, {rect, label, value, concentration});
    } else { value.textContent = `${airflow.rooms.reduce((total, room) => total + (name === 'supply' ? room.supply_m3_h : room.exhaust_m3_h), 0)} m³/h`; }
    graph.append(group);
  }
  const maximum = Math.max(...airflow.frames.flatMap(frame => frame.counts.map((bins, index) => (bins[0] + bins[1]) / airflow.rooms[index].volume_m3)));
  const slider = $('#time'); slider.max = String(airflow.frames.length - 1);
  const paint = () => {
    const frame = airflow.frames[Number(slider.value)];
    $('#time-label').textContent = `${(frame.time_h * 60).toFixed(1)} min`;
    for (const [index, room] of airflow.rooms.entries()) {
      const node = nodeValues.get(room.name);
      const concentration = (frame.counts[index][0] + frame.counts[index][1]) / room.volume_m3;
      const fraction = Math.log1p(concentration) / Math.log1p(maximum);
      node.rect.setAttribute('fill', `hsl(170 45% ${87 - fraction * 50}%)`);
      const textColor = fraction > .65 ? '#ffffff' : '#173b40';
      for (const text of [node.label, node.value, node.concentration]) text.setAttribute('fill', textColor);
      node.value.textContent = `${room.pressure_pa} Pa`;
      node.concentration.textContent = `${concentration.toFixed(1)} /m³`;
    }
  };
  $('#balance').textContent = `Particle balance residual by disjoint size bin: ${airflow.conservation_residual_by_bin.map(value => value.toExponential(2)).join(', ')} particles. Node height is decorative; link width encodes airflow.`;
  let timer;
  const stop = () => { clearInterval(timer); timer = undefined; $('#play').textContent = 'Play recorded simulation'; };
  const motion = matchMedia('(prefers-reduced-motion: reduce)');
  const applyPreference = () => { stop(); $('#play').disabled = motion.matches; };
  motion.addEventListener('change', applyPreference); applyPreference();
  $('#play').addEventListener('click', () => {
    if (timer) { stop(); return; }
    if (Number(slider.value) === Number(slider.max)) slider.value = '0';
    $('#play').textContent = 'Pause simulation';
    timer = setInterval(() => { slider.value = String(Math.min(Number(slider.max), Number(slider.value) + Number($('#speed').value))); paint(); if (slider.value === slider.max) stop(); }, 150);
  });
  slider.addEventListener('input', () => { stop(); paint(); });
  $('#reset').addEventListener('click', () => { stop(); slider.value = '0'; paint(); });
  paint();
  let offset = 0;
  async function showObservations() {
    const selected = $('#filter').value;
    const response = live ? await fetchJSON(`/api/observations?limit=12&offset=${offset}&status=${selected}`) : (() => {
      const rows = snapshot.observations.filter(row => !selected || row.status === selected);
      return {total: rows.length, rows: rows.slice(offset, offset + 12)};
    })();
    $('#observations').replaceChildren();
    for (const row of response.rows) {
      const line = element('tr');
      const normalized = row.scaled_value === null ? 'Not normalized' : `${row.scaled_value / row.value_scale} ${row.normalized_unit}`;
      for (const value of [row.source_event_id, row.declared_sensor, row.observed_at, normalized, row.status, JSON.parse(row.reasons).join('; ') || 'None']) line.append(element('td', value));
      const cell = element('td'), button = element('button', `#${row.recorded_event}`);
      button.addEventListener('click', () => inspectEvent(row.recorded_event).catch(error => status(error.message)));
      cell.append(button); line.append(cell); $('#observations').append(line);
    }
    $('#page-label').textContent = `${response.total ? offset + 1 : 0}–${Math.min(offset + 12, response.total)} of ${response.total}`;
    $('#previous').disabled = offset === 0; $('#next').disabled = offset + 12 >= response.total;
  }
  $('#filter').addEventListener('change', () => { offset = 0; showObservations().catch(error => status(error.message)); });
  $('#previous').addEventListener('click', () => { offset = Math.max(0, offset - 12); showObservations().catch(error => status(error.message)); });
  $('#next').addEventListener('click', () => { offset += 12; showObservations().catch(error => status(error.message)); });
  await inspectAsset('BR-01'); await showObservations();
}
initialize().catch(error => status(error.message));
