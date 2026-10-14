import { useEffect, useMemo, useState } from "react";
import { getHealth, predictAll } from "./api";

const initialForm = {
  range: {
    odometer_km: 55000,
    driving_mode: "highway",
    battery_pct: 65,
    battery_health_pct: 94,
    max_range_km: 548,
    energy_consumed_kwh: 8.2,
  },
  battery_degradation: {
    odometer_km: 72000,
    battery_cycles: 650,
    battery_health_pct: 91,
    battery_temperature_c: 34,
    battery_capacity_kwh: 60.5,
    energy_consumed_kwh: 10.5,
    avg_speed_kmph: 48,
    soc_start_pct: 72,
    soc_end_pct: 48,
    vehicle_age_months: 36,
  },
  rul: {
    odometer_km: 98000,
    battery_health_pct: 88,
    brake_wear_pct: 42,
    suspension_health_pct: 78,
    days_since_last_service: 145,
    vehicle_age_months: 60,
    overspeed_events: 2,
    hard_braking_events: 2,
    rapid_acceleration_events: 1,
    idle_time_minutes: 12,
    driver_experience_years: 4,
    driver_score: 72,
  },
};

const tabs = [
  { key: "overview", label: "Fleet overview", icon: "⌂" },
  { key: "range", label: "Range intelligence", icon: "◒" },
  { key: "battery", label: "Battery health", icon: "▣" },
  { key: "rul", label: "Maintenance forecast", icon: "⌁" },
];

function formatValue(value, digits = 1) {
  return typeof value === "number" ? value.toFixed(digits) : "—";
}

function App() {
  const [activeTab, setActiveTab] = useState("overview");
  const [form, setForm] = useState(initialForm);
  const [result, setResult] = useState(null);
  const [health, setHealth] = useState(null);
  const [loading, setLoading] = useState(false);
  const [healthError, setHealthError] = useState("");
  const [error, setError] = useState("");
  const [lastUpdated, setLastUpdated] = useState(null);

  useEffect(() => {
    getHealth()
      .then(setHealth)
      .catch((err) => setHealthError(err.message));
  }, []);

  const summary = useMemo(() => {
    if (!result) {
      return {
        range: "—",
        degradation: "—",
        rul: "—",
        status: "Awaiting analysis",
      };
    }
    const range = result.range?.range_km_adjusted;
    const degradation = result.battery_degradation?.battery_degradation_pct_per_100_cycles;
    const rul = result.rul?.rul_days;
    return {
      range: range == null ? "—" : `${formatValue(range, 0)} km`,
      degradation: degradation == null ? "—" : `${formatValue(degradation, 2)}%`,
      rul: rul == null ? "—" : `${formatValue(rul, 0)} days`,
      status: "Analysis complete",
    };
  }, [result]);

  function updateField(section, field, value) {
    setForm((current) => ({
      ...current,
      [section]: { ...current[section], [field]: value },
    }));
  }

  async function runAnalysis() {
    setError("");
    setLoading(true);
    try {
      const response = await predictAll(form);
      setResult(response);
      setLastUpdated(new Date());
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  function resetForm() {
    setForm(initialForm);
    setResult(null);
    setError("");
  }

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="brand">
          <div className="brand-mark"><span>⚡</span></div>
          <div>
            <div className="brand-name">EV<span>FLEET</span></div>
            <div className="brand-caption">PREDICTIVE COMMAND CENTER</div>
          </div>
        </div>
        <div className="topbar-actions">
          <div className={`connection-pill ${health ? "online" : ""}`}>
            <span className="pulse-dot" />
            {health ? "API CONNECTED" : "CONNECTING"}
          </div>
          <button className="icon-button" title="Refresh health" onClick={() => getHealth().then(setHealth).catch((err) => setHealthError(err.message))}>↻</button>
          <div className="avatar">OP</div>
        </div>
      </header>

      <div className="layout">
        <aside className="sidebar">
          <div className="sidebar-section-label">WORKSPACE</div>
          <nav>
            {tabs.map((tab) => (
              <button
                key={tab.key}
                className={`nav-item ${activeTab === tab.key ? "active" : ""}`}
                onClick={() => setActiveTab(tab.key)}
              >
                <span className="nav-icon">{tab.icon}</span>
                {tab.label}
                {tab.key === "overview" && <span className="nav-arrow">›</span>}
              </button>
            ))}
          </nav>
          <div className="sidebar-spacer" />
          <div className="model-status-card">
            <div className="sidebar-section-label">MODEL STATUS</div>
            <div className="model-status-row"><span className="status-check">✓</span><span>Range model</span><b>{health?.models_loaded?.range || "—"}</b></div>
            <div className="model-status-row"><span className="status-check">✓</span><span>Battery model</span><b>{health?.models_loaded?.battery_degradation || "—"}</b></div>
            <div className="model-status-row"><span className="status-check">✓</span><span>RUL model</span><b>{health?.models_loaded?.rul || "—"}</b></div>
          </div>
          <div className="sidebar-footer"><span className="status-check">✓</span> All systems operational</div>
        </aside>

        <main className="main-content">
          <section className="page-heading">
            <div>
              <div className="eyebrow">FLEET INTELLIGENCE / LIVE WORKSPACE</div>
              <h1>{activeTab === "overview" ? "Good morning, Operator" : tabs.find((tab) => tab.key === activeTab)?.label}</h1>
              <p>Monitor vehicle readiness, anticipate maintenance, and make every trip more predictable.</p>
            </div>
            <div className="heading-meta">
              <span className="live-indicator"><span className="pulse-dot" /> LIVE ANALYTICS</span>
              <span className="timestamp">{lastUpdated ? `Updated ${lastUpdated.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}` : "Ready for analysis"}</span>
            </div>
          </section>

          <section className="kpi-grid">
            <KpiCard label="PROJECTED RANGE" value={summary.range} detail="Physics-constrained estimate" tone="blue" icon="◒" />
            <KpiCard label="BATTERY DEGRADATION" value={summary.degradation} detail="Per 100 charge cycles" tone="green" icon="▣" />
            <KpiCard label="REMAINING USEFUL LIFE" value={summary.rul} detail="Until next major service" tone="purple" icon="⌁" />
            <KpiCard label="SYSTEM STATUS" value={summary.status} detail={healthError || "All prediction models loaded"} tone="orange" icon="✓" compact />
          </section>

          {error && <div className="alert error-alert"><strong>Prediction unavailable</strong><span>{error}</span><button onClick={() => setError("")}>×</button></div>}
          {healthError && <div className="alert warning-alert"><strong>Backend offline</strong><span>{healthError}. Start the FastAPI service on port 8000.</span></div>}

          <section className="workspace-grid">
            <div className="panel vehicle-panel">
              <div className="panel-header">
                <div><div className="eyebrow">VEHICLE VISUALIZATION</div><h2>EV-204 · Fleet vehicle</h2></div>
                <span className="vehicle-badge">ACTIVE</span>
              </div>
              <div className="scene">
                <div className="scene-grid" />
                <div className="vehicle-3d" aria-label="Animated electric vehicle">
                  <div className="car-roof" />
                  <div className="car-cabin"><span className="window front" /><span className="window rear" /></div>
                  <div className="car-body"><span className="headlight" /><span className="taillight" /><span className="door-line" /></div>
                  <div className="wheel wheel-left" /><div className="wheel wheel-right" />
                  <div className="car-glow" />
                </div>
                <div className="scene-label"><span className="pulse-dot" /> SIMULATION ACTIVE</div>
              </div>
              <div className="vehicle-stats">
                <Stat label="Battery SOC" value={`${form.range.battery_pct}%`} progress={form.range.battery_pct} color="blue" />
                <Stat label="Battery health" value={`${form.range.battery_health_pct}%`} progress={form.range.battery_health_pct} color="green" />
                <Stat label="Odometer" value={`${Number(form.range.odometer_km).toLocaleString()} km`} />
              </div>
            </div>

            <div className="panel analysis-panel">
              <div className="panel-header">
                <div><div className="eyebrow">PREDICTIVE ANALYSIS</div><h2>Run fleet assessment</h2></div>
                <span className="model-chip">3 MODELS</span>
              </div>
              <div className="analysis-tabs">
                {["overview", "range", "battery", "rul"].map((key) => (
                  <button key={key} className={activeTab === key ? "selected" : ""} onClick={() => setActiveTab(key)}>
                    {key === "overview" ? "All objectives" : tabs.find((tab) => tab.key === key)?.label}
                  </button>
                ))}
              </div>
              <div className="form-scroll">
                {(activeTab === "overview" || activeTab === "range") && <RangeForm values={form.range} onChange={(field, value) => updateField("range", field, value)} />}
                {(activeTab === "overview" || activeTab === "battery") && <BatteryForm values={form.battery_degradation} onChange={(field, value) => updateField("battery_degradation", field, value)} />}
                {(activeTab === "overview" || activeTab === "rul") && <RulForm values={form.rul} onChange={(field, value) => updateField("rul", field, value)} />}
              </div>
              <div className="form-actions">
                <button className="secondary-button" onClick={resetForm}>Reset inputs</button>
                <button className="primary-button" onClick={runAnalysis} disabled={loading || !health}>
                  {loading ? <><span className="spinner" /> Running models…</> : <>Run full assessment <span>→</span></>}
                </button>
              </div>
            </div>
          </section>

          <Results result={result} />
          <footer className="main-footer">EV Fleet Prediction API · FastAPI + scikit-learn · Decision support, not a safety certification</footer>
        </main>
      </div>
    </div>
  );
}

function KpiCard({ label, value, detail, tone, icon, compact }) {
  return <div className={`kpi-card ${tone}`}><div className="kpi-top"><span>{label}</span><b>{icon}</b></div><div className={`kpi-value ${compact ? "compact" : ""}`}>{value}</div><div className="kpi-detail">{detail}</div></div>;
}

function Stat({ label, value, progress, color }) {
  return <div className="stat"><div className="stat-label">{label}</div><strong>{value}</strong>{progress != null && <div className="progress-track"><span className={color} style={{ width: `${progress}%` }} /></div>}</div>;
}

function Field({ label, name, value, onChange, min = 0, max, step = "any", suffix }) {
  return <label className="field"><span>{label}</span><div className="input-wrap"><input type="number" name={name} min={min} max={max} step={step} value={value} onChange={(event) => onChange(name, Number(event.target.value))} />{suffix && <em>{suffix}</em>}</div></label>;
}

function SectionTitle({ icon, title, description }) {
  return <div className="form-section-title"><span className="section-icon">{icon}</span><div><h3>{title}</h3><p>{description}</p></div></div>;
}

function RangeForm({ values, onChange }) {
  return <div className="form-section"><SectionTitle icon="◒" title="Range intelligence" description="Estimate usable distance from battery state and driving conditions." /><div className="field-grid"><Field label="Odometer" name="odometer_km" value={values.odometer_km} onChange={onChange} suffix="km" /><label className="field"><span>Driving mode</span><select value={values.driving_mode} onChange={(event) => onChange("driving_mode", event.target.value)}><option value="city">City</option><option value="highway">Highway</option></select></label><Field label="Battery SOC" name="battery_pct" value={values.battery_pct} onChange={onChange} max={100} suffix="%" /><Field label="Battery health" name="battery_health_pct" value={values.battery_health_pct} onChange={onChange} max={100} suffix="%" /><Field label="Maximum range" name="max_range_km" value={values.max_range_km} onChange={onChange} min={0.01} suffix="km" /><Field label="Energy consumed" name="energy_consumed_kwh" value={values.energy_consumed_kwh} onChange={onChange} suffix="kWh" /></div></div>;
}

function BatteryForm({ values, onChange }) {
  return <div className="form-section"><SectionTitle icon="▣" title="Battery degradation" description="Measure expected battery health loss per 100 cycles." /><div className="field-grid"><Field label="Odometer" name="odometer_km" value={values.odometer_km} onChange={onChange} suffix="km" /><Field label="Battery cycles" name="battery_cycles" value={values.battery_cycles} onChange={onChange} suffix="cycles" /><Field label="Battery health" name="battery_health_pct" value={values.battery_health_pct} onChange={onChange} max={100} suffix="%" /><Field label="Temperature" name="battery_temperature_c" value={values.battery_temperature_c} onChange={onChange} suffix="°C" /><Field label="Pack capacity" name="battery_capacity_kwh" value={values.battery_capacity_kwh} onChange={onChange} min={0.01} suffix="kWh" /><Field label="Energy consumed" name="energy_consumed_kwh" value={values.energy_consumed_kwh} onChange={onChange} suffix="kWh" /><Field label="Average speed" name="avg_speed_kmph" value={values.avg_speed_kmph} onChange={onChange} suffix="km/h" /><Field label="SOC at start" name="soc_start_pct" value={values.soc_start_pct} onChange={onChange} max={100} suffix="%" /><Field label="SOC at end" name="soc_end_pct" value={values.soc_end_pct} onChange={onChange} max={100} suffix="%" /><Field label="Vehicle age" name="vehicle_age_months" value={values.vehicle_age_months} onChange={onChange} suffix="months" /></div></div>;
}

function RulForm({ values, onChange }) {
  return <div className="form-section"><SectionTitle icon="⌁" title="Maintenance forecast" description="Estimate remaining useful life from vehicle, driver, and service signals." /><div className="field-grid"><Field label="Odometer" name="odometer_km" value={values.odometer_km} onChange={onChange} suffix="km" /><Field label="Battery health" name="battery_health_pct" value={values.battery_health_pct} onChange={onChange} max={100} suffix="%" /><Field label="Brake wear" name="brake_wear_pct" value={values.brake_wear_pct} onChange={onChange} max={100} suffix="%" /><Field label="Suspension health" name="suspension_health_pct" value={values.suspension_health_pct} onChange={onChange} max={100} suffix="%" /><Field label="Days since service" name="days_since_last_service" value={values.days_since_last_service} onChange={onChange} suffix="days" /><Field label="Vehicle age" name="vehicle_age_months" value={values.vehicle_age_months} onChange={onChange} suffix="months" /><Field label="Overspeed events" name="overspeed_events" value={values.overspeed_events} onChange={onChange} /><Field label="Hard braking events" name="hard_braking_events" value={values.hard_braking_events} onChange={onChange} /><Field label="Rapid acceleration" name="rapid_acceleration_events" value={values.rapid_acceleration_events} onChange={onChange} /><Field label="Idle time" name="idle_time_minutes" value={values.idle_time_minutes} onChange={onChange} suffix="min" /><Field label="Driver experience" name="driver_experience_years" value={values.driver_experience_years} onChange={onChange} suffix="years" /><Field label="Driver score" name="driver_score" value={values.driver_score} onChange={onChange} max={100} suffix="/100" /></div></div>;
}

function Results({ result }) {
  if (!result) return <section className="empty-results"><div className="empty-icon">✦</div><div><h2>Your prediction workspace is ready</h2><p>Enter operating conditions above and run the full assessment to see model outputs here.</p></div></section>;
  return <section className="results-section"><div className="result-heading"><div><div className="eyebrow">LATEST ASSESSMENT</div><h2>Fleet decision signals</h2></div><span className="result-time">✓ Verified response from API</span></div><div className="result-grid"><ResultCard tone="blue" icon="◒" title="Projected range" value={`${formatValue(result.range.range_km_adjusted, 1)} km`} note={`Physical bound: ${formatValue(result.range.range_km_physical, 1)} km`} model={result.range.model} /><ResultCard tone="green" icon="▣" title="Battery degradation" value={`${formatValue(result.battery_degradation.battery_degradation_pct_per_100_cycles, 3)}%`} note={`SOC drop: ${formatValue(result.battery_degradation.soc_drop_pct, 1)}%`} model={result.battery_degradation.model} /><ResultCard tone="purple" icon="⌁" title="Remaining useful life" value={`${formatValue(result.rul.rul_days, 1)} days`} note={`Mechanical health: ${formatValue(result.rul.mechanical_health, 1)}/100`} model={result.rul.model} /></div></section>;
}

function ResultCard({ tone, icon, title, value, note, model }) {
  return <div className={`result-card ${tone}`}><div className="result-card-top"><span className="result-icon">{icon}</span><span className="model-label">{model}</span></div><div className="result-title">{title}</div><div className="result-value">{value}</div><div className="result-note">{note}</div></div>;
}

export default App;
