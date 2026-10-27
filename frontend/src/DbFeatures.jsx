import React, { useCallback, useEffect, useState } from 'react';

/* ------------------------------------------------------------------
   Database features page — Part A (UDFs / UDA) and Part B (NoSQL-native).
   Every card calls the FastAPI adapter (db_api.py), which calls the
   RaceStore facade, which runs the CQL shown in the card.
------------------------------------------------------------------- */

const FEATURES = [
  // ---- Part A ----------------------------------------------------
  { key: 'urgency_score_udf', part: 'A', n: 1, tag: 'UDF', title: 'urgency_score_udf',
    what: 'Computes a 0–1 urgency score server-side directly from the gap, tyre-wear and weather columns for a given lap, so the score never needs to be calculated in application code. Shown as computed_urgency in the enriched lap read.' },
  { key: 'pit_window_flag', part: 'A', n: 2, tag: 'UDF', title: 'pit_window_flag',
    what: 'Flags whether a driver’s current tyre wear has crossed the pit-window threshold, returning a simple true/false in_pit_window computed inline by the database.' },
  { key: 'gap_threat_level', part: 'A', n: 3, tag: 'UDF', title: 'gap_threat_level',
    what: 'Classifies defensive pressure (under_attack / pressure / safe) from the gap to the cars ahead and behind, done as part of the query itself in one round trip.' },
  { key: 'response_time_bucket', part: 'A', n: 4, tag: 'UDF', title: 'response_time_bucket',
    what: 'Buckets a prediction’s latency into fast / medium / slow directly from predictions_log, so latency reporting is server-side rather than post-processed in Python.' },
  { key: 'avg_urgency_by_race', part: 'A', n: 5, tag: 'UDA', title: 'avg_urgency_by_race',
    what: 'A true User-Defined Aggregate: folds every row in a partition into a running average of urgency_score in a single server-side pass, instead of pulling all rows back to average them in the client.' },
  // ---- Part B ----------------------------------------------------
  { key: 'ttl', part: 'B', n: 1, tag: 'TTL', title: 'TTL (self-expiring rows)',
    what: 'Rows in live_leaderboard are written with a time-to-live and delete themselves after 10 minutes, with no cleanup job required.',
    vs: 'Relational: needs a scheduled DELETE job.' },
  { key: 'counters', part: 'B', n: 2, tag: 'COUNTER', title: 'Counter columns',
    what: 'Atomic, race-free increments for event counts (predictions_fast_local, predictions_smart_cloud, pit_stops), avoiding the read-modify-write pattern a normal column needs under concurrent updates.',
    vs: 'Relational: needs row locks or an atomic-update workaround.' },
  { key: 'collections', part: 'B', n: 3, tag: 'SET + LIST', title: 'Collections (SET + LIST)',
    what: 'Nested, denormalised data — e.g. the set of tyre compounds a driver used — lives in a single row instead of a join table. Duplicates added to a SET (SOFT twice) are dropped automatically.',
    vs: 'Relational: a separate join table.' },
  { key: 'materialized_view', part: 'B', n: 4, tag: 'MV', title: 'Materialized view',
    what: 'Cassandra maintains a second, differently-keyed copy of predictions_log, so rows can be looked up by driver alone (without run_id or race_id) with no manual syncing.',
    vs: 'Relational: triggers or application-side double writes.' },
  { key: 'lwt', part: 'B', n: 5, tag: 'LWT', title: 'Lightweight transaction (Paxos CAS)',
    what: 'A real linearizable guarantee: insert a row only if it doesn’t already exist. The first insert for a (race, driver) returns True (created); a repeat returns False (already exists).',
    vs: 'Default Cassandra writes silently overwrite; LWT is opt-in.' },
];

/* Clearly-labelled illustrative output, shown only if the user asks for it
   while no Cassandra is connected. Not produced by a live database. */
const SAMPLE = {
  urgency_score_udf: { cql: 'SELECT urgency_score_udf(gap_ahead, gap_behind, tyre_life, is_caution,\n       rainfall, field_pits_last_5) AS computed_urgency\nFROM telemetry_laps WHERE race_id=? AND driver=? AND lap_number=?', result: { inputs: { gap_ahead: 0.77, gap_behind: 0.55, tyre_life: 22, is_caution: 0, rainfall: false, field_pits_last_5: 3 }, computed_urgency: 0.75, python_urgency: 0.75, matches_python: true } },
  pit_window_flag: { cql: 'SELECT tyre_life, pit_window_flag(tyre_life) AS in_pit_window\nFROM telemetry_laps WHERE ...', result: { tyre_life: 22, in_pit_window: true, rule: 'tyre_life >= 20 laps' } },
  gap_threat_level: { cql: 'SELECT gap_ahead, gap_behind, gap_threat_level(gap_ahead, gap_behind) AS threat_level\nFROM telemetry_laps WHERE ...', result: { gap_ahead: 0.77, gap_behind: 0.55, threat_level: 'under_attack', rule: 'behind<1s: under_attack · behind<2s or ahead<1s: pressure · else safe' } },
  response_time_bucket: { cql: 'SELECT response_time_bucket(response_time_ms) AS bucket\nFROM predictions_log WHERE run_id=? AND race_id=?', result: { run_id: 'sample', bucket_counts: { fast: 2, medium: 1, slow: 1, unknown: 0 }, rule: '<10 ms fast · <60 ms medium · else slow' } },
  avg_urgency_by_race: { cql: 'SELECT avg_urgency_by_race(urgency_score) AS avg_urgency\nFROM predictions_log WHERE run_id=? AND race_id=?', result: { run_id: 'sample', server_side_avg: 0.5925, rows_folded: 4, client_side_check: 0.5925, matches: true } },
  ttl: { cql: 'INSERT INTO live_leaderboard (...) VALUES (...) USING TTL 600;\nSELECT driver, lap_number, urgency, TTL(urgency) FROM live_leaderboard WHERE race_id=?', result: { ttl_seconds_at_write: 600, rows: [{ driver: 'VER', lap_number: 23, urgency: 0.9, ttl_remaining_sec: 598 }] } },
  counters: { cql: 'UPDATE race_event_counters SET count = count + 1\nWHERE race_id=? AND counter_name=?', result: { event_counters: { predictions_fast_local: 2, predictions_smart_cloud: 2, pit_stops: 2 }, concurrency_test: { parallel_increments: 20, before: 0, after: 20, expected: 20, no_lost_updates: true } } },
  collections: { cql: "UPDATE driver_race_summary SET compounds_used = compounds_used + {'SOFT'} ...;\nUPDATE driver_race_summary SET stint_history = stint_history + ['MEDIUM:1-20'] ...", result: { 'compounds_used (SET)': ['MEDIUM', 'SOFT'], 'stint_history (LIST)': ['MEDIUM:1-20'], note: 'SOFT was added twice; the SET keeps one copy.' } },
  materialized_view: { cql: 'SELECT * FROM predictions_by_driver WHERE driver=? LIMIT 5', result: { driver: 'VER', rows_found: 2, rows: [{ race_id: 'sample', lap_number: 23, urgency_score: 0.9, model_used: 'smart_cloud', response_time_ms: 96.5 }, { race_id: 'sample', lap_number: 21, urgency_score: 0.55, model_used: 'fast_local', response_time_ms: 5.1 }] } },
  lwt: { cql: 'INSERT INTO race_drivers (race_id, driver, max_lap)\nVALUES (?, ?, ?) IF NOT EXISTS', result: { first_insert_applied: true, second_insert_applied: false, expected: 'True then False', real_row_applied: false, real_row_note: 'False means the dataset loader had already created this (race, driver) row — LWT refused to overwrite it.' } },
};

const num = (v, d = 3) => (v === null || v === undefined || Number.isNaN(Number(v)) ? '—' : Number(v).toLocaleString(undefined, { maximumFractionDigits: d }));
const mmss = (s) => `${Math.floor(s / 60)}:${String(Math.max(0, s % 60)).padStart(2, '0')}`;

async function call(API, path, options) {
  const res = await fetch(`${API}${path}`, options);
  let body = null;
  try { body = await res.json(); } catch { /* non-JSON error body */ }
  if (!res.ok) throw new Error(body?.detail || `Request failed (${res.status})`);
  return body;
}
const post = (body) => ({ method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });

/* ---------------------------- result views ------------------------- */
function Chip({ ok, children }) {
  return <span className={`db-chip ${ok === true ? 'ok' : ok === false ? 'bad' : ''}`}>{children}</span>;
}
function KV({ k, v }) {
  return <div className="db-kv"><span>{k}</span><b>{v}</b></div>;
}

function ResultView({ fkey, r }) {
  switch (fkey) {
    case 'urgency_score_udf': return (
      <>
        <div className="db-big">{num(r.computed_urgency, 2)}<small>computed_urgency · 0–1</small></div>
        <div className="db-meter"><i style={{ width: `${Math.min(1, r.computed_urgency || 0) * 100}%` }} /></div>
        <div className="db-kvs">{Object.entries(r.inputs || {}).map(([k, v]) => <KV key={k} k={k} v={String(v ?? '—')} />)}</div>
        {r.matches_python !== undefined && <p className="db-line"><Chip ok={r.matches_python}>{r.matches_python ? 'MATCHES' : 'DIFFERS FROM'} PYTHON</Chip> sched/urgency.py scored {num(r.python_urgency, 2)} for the same lap.</p>}
        {r.parity_check && <p className="db-line muted">Python parity check: {r.parity_check}</p>}
      </>);
    case 'pit_window_flag': return (
      <>
        <div className="db-big">{String(r.in_pit_window)}<small>in_pit_window</small></div>
        <div className="db-kvs"><KV k="tyre_life" v={`${num(r.tyre_life, 1)} laps`} /><KV k="rule" v={r.rule} /></div>
      </>);
    case 'gap_threat_level': return (
      <>
        <div className="db-levels">{['under_attack', 'pressure', 'safe'].map(l => <span key={l} className={`db-level ${l} ${r.threat_level === l ? 'on' : ''}`}>{l.replace('_', ' ')}</span>)}</div>
        <div className="db-kvs"><KV k="gap_ahead" v={`${num(r.gap_ahead, 2)} s`} /><KV k="gap_behind" v={`${num(r.gap_behind, 2)} s`} /></div>
        <p className="db-line muted">{r.rule}</p>
      </>);
    case 'response_time_bucket': {
      const c = r.bucket_counts || {}; const max = Math.max(1, ...Object.values(c));
      return (
        <>
          <div className="db-bars">{['fast', 'medium', 'slow'].map(k => <div key={k}><span>{k}</span><div className="db-bar"><i className={k} style={{ width: `${(c[k] || 0) / max * 100}%` }} /></div><b>{c[k] || 0}</b></div>)}</div>
          <p className="db-line muted">{r.rule} · run {r.run_id}</p>
        </>);
    }
    case 'avg_urgency_by_race': return (
      <>
        <div className="db-big">{num(r.server_side_avg, 3)}<small>server-side average</small></div>
        <div className="db-kvs"><KV k="rows folded in Cassandra" v={r.rows_folded} /><KV k="client-side recompute" v={num(r.client_side_check, 3)} /></div>
        <p className="db-line"><Chip ok={r.matches}>{r.matches ? 'AGREES' : 'DISAGREES'}</Chip> with a Python average over the same rows (check only; the UDA needs no rows sent back).</p>
      </>);
    case 'ttl': return <TtlView r={r} />;
    case 'counters': {
      const t = r.concurrency_test || {};
      return (
        <>
          <div className="db-tiles">{Object.entries(r.event_counters || {}).map(([k, v]) => <div key={k}><b>{v}</b><small>{k}</small></div>)}</div>
          <p className="db-line"><Chip ok={t.no_lost_updates}>{t.no_lost_updates ? 'NO LOST UPDATES' : 'LOST UPDATES'}</Chip> {t.parallel_increments} parallel increments: {t.before} → {t.after} (expected {t.expected}).</p>
        </>);
    }
    case 'collections': return (
      <>
        <div className="db-kvs wide"><KV k="compounds_used · SET" v={<span className="db-chips">{(r['compounds_used (SET)'] || []).map(c => <Chip key={c}>{c}</Chip>)}</span>} />
          <KV k="stint_history · LIST" v={<ol className="db-list">{(r['stint_history (LIST)'] || []).map((s, i) => <li key={i}>{s}</li>)}</ol>} /></div>
        <p className="db-line muted">{r.note}</p>
      </>);
    case 'materialized_view': return (
      <>
        <p className="db-line"><b>{r.rows_found}</b> prediction row(s) for driver <b>{r.driver}</b>, no run_id / race_id supplied.</p>
        {r.rows?.length > 0 && <div className="db-table-wrap"><table className="db-table"><thead><tr><th>race</th><th>lap</th><th>urgency</th><th>model</th><th>ms</th></tr></thead>
          <tbody>{r.rows.map((x, i) => <tr key={i}><td>{String(x.race_id).replaceAll('_', ' ')}</td><td>{x.lap_number}</td><td>{num(x.urgency_score, 2)}</td><td>{x.model_used}</td><td>{num(x.response_time_ms, 1)}</td></tr>)}</tbody></table></div>}
      </>);
    case 'lwt': return (
      <>
        <div className="db-steps">
          <div><small>1st insert</small><Chip ok={r.first_insert_applied}>{String(r.first_insert_applied)}</Chip><em>created</em></div>
          <span>→</span>
          <div><small>2nd insert</small><Chip ok={r.second_insert_applied === false}>{String(r.second_insert_applied)}</Chip><em>already exists</em></div>
        </div>
        <p className="db-line muted">Expected: {r.expected}. On the real row: <b>{String(r.real_row_applied)}</b> — {r.real_row_note}</p>
      </>);
    default: return <pre className="db-raw">{JSON.stringify(r, null, 2)}</pre>;
  }
}

function TtlView({ r }) {
  const [elapsed, setElapsed] = useState(0);
  useEffect(() => { setElapsed(0); const t = setInterval(() => setElapsed(e => e + 1), 1000); return () => clearInterval(t); }, [r]);
  const rows = r.rows || [];
  return (
    <>
      <p className="db-line">Written with <b>USING TTL {r.ttl_seconds_at_write}</b> ({mmss(r.ttl_seconds_at_write)}). No cleanup job exists; Cassandra drops each row itself.</p>
      {rows.length > 0 ? <div className="db-table-wrap"><table className="db-table"><thead><tr><th>driver</th><th>lap</th><th>urgency</th><th>expires in</th></tr></thead>
        <tbody>{rows.map(x => { const left = Math.max(0, Math.round((x.ttl_remaining_sec ?? 0) - elapsed)); return <tr key={x.driver}><td>{x.driver}</td><td>{x.lap_number}</td><td>{num(x.urgency, 2)}</td><td className="db-mono">{left > 0 ? mmss(left) : 'expired'}</td></tr>; })}</tbody></table></div>
        : <p className="db-line muted">No live rows — they have expired (or none were written yet). Press Run again to refill.</p>}
    </>);
}

/* ------------------------------ card ------------------------------- */
function FeatureCard({ f, entry, running, onRun, sample }) {
  const [showCql, setShowCql] = useState(false);
  const state = running ? 'running' : entry ? (entry.ok ? 'done' : 'failed') : 'idle';
  return (
    <article className={`panel db-card ${state}`}>
      <header>
        <span className="db-num">{f.part}{f.n}</span>
        <div><h3>{f.title}</h3><span className="db-tag">{f.tag}</span></div>
        <button className="secondary-button db-run" onClick={onRun} disabled={running || sample}>{running ? <><span className="spinner" /> Running</> : entry ? '↻ Re-run' : '▶ Run'}</button>
      </header>
      <p className="db-what">{f.what}</p>
      {f.vs && <p className="db-vs"><b>vs relational</b> {f.vs.replace(/^Relational: /, '')}</p>}
      <div className="db-result">
        {!entry && !running && <p className="db-empty">Not run yet.</p>}
        {entry && entry.ok && <><ResultView fkey={f.key} r={entry.result} />
          <div className="db-foot"><button className="text-button" onClick={() => setShowCql(s => !s)}>{showCql ? 'Hide' : 'Show'} CQL ↴</button>{!sample && <span>{entry.elapsed_ms} ms round trip</span>}</div>
          {showCql && <pre className="db-cql">{entry.cql}</pre>}</>}
        {entry && !entry.ok && <div className="db-error"><b>Failed</b><p>{entry.error}</p>{entry.hint && <p className="db-hint">{entry.hint}</p>}</div>}
      </div>
    </article>
  );
}

/* ------------------------------ page ------------------------------- */
export default function DbFeatures({ API, health }) {
  const [status, setStatus] = useState(null);       // {connected, error}
  const [opts, setOpts] = useState(null);           // races, drivers, laps
  const [race, setRace] = useState('');
  const [driver, setDriver] = useState('');
  const [lap, setLap] = useState(20);
  const [results, setResults] = useState({});
  const [busy, setBusy] = useState({});             // key -> bool, or '*' for all
  const [error, setError] = useState('');
  const [sample, setSample] = useState(false);

  const loadStatus = useCallback(() =>
    call(API, '/api/db/status').then(setStatus).catch(e => setStatus({ connected: false, error: e.message })), [API]);
  const loadOptions = useCallback((r) =>
    call(API, `/api/db/options${r ? `?race=${encodeURIComponent(r)}` : ''}`).then(o => {
      setOpts(o); setRace(o.race);
      setDriver(d => (o.drivers.includes(d) ? d : (o.drivers.includes('VER') ? 'VER' : o.drivers[0] || '')));
      setLap(l => Math.min(Math.max(l, o.min_lap), o.max_lap));
    }).catch(e => setError(e.message)), [API]);

  useEffect(() => { if (health === 'online') { loadStatus(); loadOptions(); } }, [health, loadStatus, loadOptions]);

  const req = { race, driver, lap: Number(lap) };
  const ready = status?.connected && race && driver && !Number.isNaN(req.lap);

  const runOne = async (key) => {
    setError(''); setBusy(b => ({ ...b, [key]: true }));
    try { const r = await call(API, `/api/db/run/${key}`, post(req)); setResults(x => ({ ...x, [key]: r })); }
    catch (e) { setError(e.message); }
    finally { setBusy(b => ({ ...b, [key]: false })); }
  };
  const runAll = async () => {
    setError(''); setBusy({ '*': true });
    try { const r = await call(API, '/api/db/run-all', post(req)); setResults(Object.fromEntries(r.results.map(x => [x.key, x]))); }
    catch (e) { setError(e.message); }
    finally { setBusy({}); }
  };
  const reconnect = async () => { setError(''); setStatus(await call(API, '/api/db/reconnect', { method: 'POST' }).catch(e => ({ connected: false, error: e.message }))); };

  const shown = sample ? Object.fromEntries(Object.entries(SAMPLE).map(([k, v]) => [k, { ok: true, ...v }])) : results;
  const doneCount = FEATURES.filter(f => shown[f.key]?.ok).length;
  const failCount = FEATURES.filter(f => shown[f.key] && !shown[f.key].ok).length;
  const part = (p) => FEATURES.filter(f => f.part === p);
  const offline = health !== 'online';

  return (
    <section className="page db-page">
      <div className="page-title">
        <div className="eyebrow">DATABASE WORKSTREAM · CASSANDRA</div>
        <h1>Database features</h1>
        <p>Ten capabilities built into Cassandra itself: five server-side functions that move logic into the query, and five NoSQL-native features a relational table either can’t do or needs extra application code or a scheduled job to replicate. Each card runs against the live keyspace and shows the CQL behind it.</p>
      </div>

      {sample && <div className="demo-banner"><span>SAMPLE</span><div><b>Illustrative sample output — not from a live database</b><p>Shown only so the layout can be reviewed without Cassandra running. Values are hand-written to match the documented behaviour of each feature.</p></div><button onClick={() => setSample(false)}>Exit sample ×</button></div>}

      <div className="panel db-controls">
        <div className="db-conn">
          <span className={`status-dot ${offline ? 'offline' : status?.connected ? 'online' : status ? 'offline' : 'checking'}`} />
          <div><b>{offline ? 'Scheduler API offline' : status?.connected ? `Cassandra connected · keyspace ${status.keyspace}` : status ? 'Cassandra not reachable' : 'Checking Cassandra…'}</b>
            {!offline && status && !status.connected && <small>{status.error}</small>}
            {offline && <small>Start the backend: uvicorn api:app --port 8000</small>}</div>
          {!offline && status && !status.connected && <button className="secondary-button" onClick={reconnect}>Retry connection</button>}
        </div>
        <div className="db-fields">
          <div><label className="field-label">Race</label>
            <select value={race} onChange={e => { setRace(e.target.value); loadOptions(e.target.value); }} disabled={!opts}>{(opts?.races || []).map(r => <option key={r} value={r}>{r.replaceAll('_', ' ')}</option>)}</select></div>
          <div><label className="field-label">Driver</label>
            <select value={driver} onChange={e => setDriver(e.target.value)} disabled={!opts}>{(opts?.drivers || []).map(d => <option key={d}>{d}</option>)}</select></div>
          <div><label className="field-label">Lap {opts && <b>{opts.min_lap}–{opts.max_lap}</b>}</label>
            <input type="number" min={opts?.min_lap} max={opts?.max_lap} value={lap} onChange={e => setLap(e.target.value)} /></div>
          <button className="primary-button" onClick={runAll} disabled={!ready || busy['*']}>{busy['*'] ? <><span className="spinner" /> Running all…</> : '▶  Run all 10'}</button>
        </div>
        <p className="field-help">Telemetry must be loaded first: <code>python -m data.load_dataset --races {race || '<race>'}</code>. “Run all” starts a fresh run_id and logs four demo predictions so the aggregate, bucket, leaderboard, counter and view features have data.</p>
        {!offline && status && !status.connected && !sample && <button className="text-button" onClick={() => setSample(true)}>Preview layout with illustrative sample ↗</button>}
      </div>

      {error && <div className="error-banner"><b>Request failed</b><p>{error}</p></div>}

      <div className="db-summary">
        <Metric label="Features run OK" value={`${doneCount} / 10`} accent={doneCount === 10} />
        <Metric label="Failed" value={failCount} accent={false} warn={failCount > 0} />
        <Metric label="Part A · UDFs" value={`${part('A').filter(f => shown[f.key]?.ok).length} / 5`} />
        <Metric label="Part B · NoSQL" value={`${part('B').filter(f => shown[f.key]?.ok).length} / 5`} />
      </div>

      <div className="section-heading compact"><div><div className="eyebrow">PART A</div><h2>User-Defined Functions</h2></div><span className="section-note">Defined in data/schema_extensions.cql · executed inside Cassandra</span></div>
      <div className="db-grid">{part('A').map(f => <FeatureCard key={f.key} f={f} entry={shown[f.key]} running={!!busy[f.key] || !!busy['*']} sample={sample} onRun={() => runOne(f.key)} />)}</div>

      <div className="section-heading compact"><div><div className="eyebrow">PART B</div><h2>NoSQL-native features</h2></div><span className="section-note">Not available, or not natively available, in the SQLite baseline</span></div>
      <div className="db-grid">{part('B').map(f => <FeatureCard key={f.key} f={f} entry={shown[f.key]} running={!!busy[f.key] || !!busy['*']} sample={sample} onRun={() => runOne(f.key)} />)}</div>
    </section>
  );
}

function Metric({ label, value, accent, warn }) {
  return <div className={`metric-card ${accent ? 'accent' : ''} ${warn ? 'warn' : ''}`}><span>{label}</span><b>{value}</b></div>;
}
