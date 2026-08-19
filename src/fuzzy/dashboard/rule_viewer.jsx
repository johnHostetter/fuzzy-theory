import { useState, useMemo } from "react";

const TERM_COLORS = [
  "#4a90d9", "#d94a6b", "#5cb85c", "#f0ad4e", "#9b59b6",
  "#1abc9c", "#e67e22", "#e74c3c", "#3498db", "#2ecc71",
];

function generateDemoData(numVars, numTerms, numRules) {
  const tensor = [];
  for (let v = 0; v < numVars; v++) {
    const terms = [];
    for (let t = 0; t < numTerms; t++) {
      const rules = [];
      for (let r = 0; r < numRules; r++) {
        rules.push(0);
      }
      terms.push(rules);
    }
    tensor.push(terms);
  }
  for (let r = 0; r < numRules; r++) {
    const activeCount = 3 + Math.floor(Math.random() * 6);
    const chosen = new Set();
    while (chosen.size < Math.min(activeCount, numVars)) {
      chosen.add(Math.floor(Math.random() * numVars));
    }
    for (const v of chosen) {
      const t = Math.floor(Math.random() * numTerms);
      tensor[v][t][r] = 1;
    }
  }
  return tensor;
}

function tensorToRules(tensor, varNames, termNames) {
  const numVars = tensor.length;
  const numRules = tensor[0][0].length;
  const rules = [];
  for (let r = 0; r < numRules; r++) {
    const clauses = [];
    for (let v = 0; v < numVars; v++) {
      for (let t = 0; t < tensor[v].length; t++) {
        if (tensor[v][t][r]) {
          clauses.push({ var: v, varName: varNames[v], term: t, termName: termNames[t] });
        }
      }
    }
    rules.push({ id: r, clauses });
  }
  return rules;
}

function TermBadge({ termIndex, termName, total }) {
  const color = TERM_COLORS[termIndex % TERM_COLORS.length];
  return (
    <span style={{
      display: "inline-block",
      padding: "2px 8px",
      borderRadius: "4px",
      fontSize: "12px",
      fontWeight: 600,
      fontFamily: "'SF Mono', 'Fira Code', monospace",
      background: color + "20",
      color: color,
      border: `1px solid ${color}40`,
    }}>
      {termName}
    </span>
  );
}

export default function RuleViewer() {
  const numVars = 20;
  const numTerms = 5;
  const numRules = 12;

  const varNames = Array.from({ length: numVars }, (_, i) => `x${i}`);
  const termNames = ["very_low", "low", "medium", "high", "very_high"];

  const tensor = useMemo(() => generateDemoData(numVars, numTerms, numRules), []);
  const rules = useMemo(() => tensorToRules(tensor, varNames, termNames), [tensor]);

  const [view, setView] = useState("cards");
  const [search, setSearch] = useState("");
  const [expandedRule, setExpandedRule] = useState(null);
  const [filterTerm, setFilterTerm] = useState(null);

  const filtered = useMemo(() => {
    let result = rules;
    if (search.trim()) {
      const q = search.toLowerCase();
      result = result.filter(r =>
        r.clauses.some(c => c.varName.toLowerCase().includes(q) || c.termName.toLowerCase().includes(q))
      );
    }
    if (filterTerm !== null) {
      result = result.filter(r => r.clauses.some(c => c.term === filterTerm));
    }
    return result;
  }, [rules, search, filterTerm]);

  const allActiveVars = useMemo(() => {
    const s = new Set();
    rules.forEach(r => r.clauses.forEach(c => s.add(c.var)));
    return [...s].sort((a, b) => a - b);
  }, [rules]);

  return (
    <div style={{
      fontFamily: "'Inter', -apple-system, sans-serif",
      color: "var(--text-color, #e0e0e0)",
      background: "var(--bg-color, #1a1a2e)",
      minHeight: "100vh",
      padding: "24px",
    }}>
      <div style={{ maxWidth: 1100, margin: "0 auto" }}>
        <div style={{ marginBottom: 24 }}>
          <h1 style={{ fontSize: 20, fontWeight: 700, margin: "0 0 4px", letterSpacing: "-0.02em" }}>
            Fuzzy Rule Inspector
          </h1>
          <p style={{ fontSize: 13, opacity: 0.5, margin: 0 }}>
            {rules.length} rules · {numVars} variables · {numTerms} terms
          </p>
        </div>

        {/* Controls */}
        <div style={{ display: "flex", gap: 12, marginBottom: 20, flexWrap: "wrap", alignItems: "center" }}>
          <input
            type="text"
            placeholder="Filter by variable or term…"
            value={search}
            onChange={e => setSearch(e.target.value)}
            style={{
              flex: "1 1 200px",
              padding: "8px 12px",
              borderRadius: 6,
              border: "1px solid var(--border-color, #333)",
              background: "var(--input-bg, #16213e)",
              color: "inherit",
              fontSize: 13,
              outline: "none",
            }}
          />
          <div style={{ display: "flex", gap: 4 }}>
            {["cards", "matrix"].map(v => (
              <button
                key={v}
                onClick={() => setView(v)}
                style={{
                  padding: "6px 14px",
                  borderRadius: 6,
                  border: "1px solid var(--border-color, #333)",
                  background: view === v ? "var(--accent, #4a90d9)" : "transparent",
                  color: view === v ? "#fff" : "inherit",
                  cursor: "pointer",
                  fontSize: 12,
                  fontWeight: 600,
                  textTransform: "capitalize",
                }}
              >
                {v}
              </button>
            ))}
          </div>
        </div>

        {/* Term legend / filter */}
        <div style={{ display: "flex", gap: 6, marginBottom: 20, flexWrap: "wrap" }}>
          <span style={{ fontSize: 12, opacity: 0.5, lineHeight: "26px" }}>Filter:</span>
          {termNames.map((tn, i) => (
            <button
              key={i}
              onClick={() => setFilterTerm(filterTerm === i ? null : i)}
              style={{
                padding: "3px 10px",
                borderRadius: 4,
                border: filterTerm === i ? `2px solid ${TERM_COLORS[i % TERM_COLORS.length]}` : "1px solid transparent",
                background: TERM_COLORS[i % TERM_COLORS.length] + (filterTerm === i ? "30" : "15"),
                color: TERM_COLORS[i % TERM_COLORS.length],
                cursor: "pointer",
                fontSize: 12,
                fontWeight: 500,
              }}
            >
              {tn}
            </button>
          ))}
        </div>

        {/* Card View */}
        {view === "cards" && (
          <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
            {filtered.map(rule => {
              const isExpanded = expandedRule === rule.id;
              const displayClauses = isExpanded ? rule.clauses : rule.clauses.slice(0, 8);
              const overflow = rule.clauses.length - 8;
              return (
                <div
                  key={rule.id}
                  onClick={() => setExpandedRule(isExpanded ? null : rule.id)}
                  style={{
                    padding: "12px 16px",
                    borderRadius: 8,
                    border: "1px solid var(--border-color, #2a2a4a)",
                    background: "var(--card-bg, #16213e)",
                    cursor: "pointer",
                    transition: "border-color 0.15s",
                  }}
                >
                  <div style={{ display: "flex", alignItems: "baseline", gap: 12, marginBottom: 8 }}>
                    <span style={{
                      fontFamily: "'SF Mono', monospace",
                      fontSize: 11,
                      fontWeight: 700,
                      opacity: 0.4,
                      minWidth: 32,
                    }}>
                      R{rule.id}
                    </span>
                    <span style={{ fontSize: 12, opacity: 0.4 }}>
                      {rule.clauses.length} clause{rule.clauses.length !== 1 ? "s" : ""}
                    </span>
                    {!isExpanded && overflow > 0 && (
                      <span style={{ fontSize: 11, opacity: 0.35, marginLeft: "auto" }}>
                        +{overflow} more
                      </span>
                    )}
                  </div>
                  <div style={{ display: "flex", flexWrap: "wrap", gap: 6, alignItems: "center" }}>
                    {displayClauses.map((c, i) => (
                      <span key={i} style={{ display: "inline-flex", alignItems: "center", gap: 4 }}>
                        {i > 0 && <span style={{ fontSize: 10, opacity: 0.3, fontWeight: 700 }}>∧</span>}
                        <span style={{ fontSize: 12, opacity: 0.6, fontFamily: "'SF Mono', monospace" }}>
                          {c.varName}
                        </span>
                        <span style={{ fontSize: 10, opacity: 0.3 }}>=</span>
                        <TermBadge termIndex={c.term} termName={c.termName} total={numTerms} />
                      </span>
                    ))}
                    {isExpanded && (
                      <span style={{
                        display: "block",
                        width: "100%",
                        marginTop: 8,
                        fontSize: 12,
                        opacity: 0.5,
                        fontFamily: "monospace",
                        borderTop: "1px solid var(--border-color, #2a2a4a)",
                        paddingTop: 8,
                      }}>
                        → THEN ___
                      </span>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
        )}

        {/* Matrix View */}
        {view === "matrix" && (
          <div style={{ overflowX: "auto", borderRadius: 8, border: "1px solid var(--border-color, #2a2a4a)" }}>
            <table style={{ borderCollapse: "collapse", fontSize: 12, width: "100%" }}>
              <thead>
                <tr>
                  <th style={{
                    position: "sticky", left: 0, zIndex: 2,
                    padding: "8px 12px",
                    background: "var(--card-bg, #16213e)",
                    borderBottom: "2px solid var(--border-color, #2a2a4a)",
                    textAlign: "left",
                    fontFamily: "monospace",
                    fontSize: 11,
                    fontWeight: 600,
                  }}>
                    Rule
                  </th>
                  {allActiveVars.map(v => (
                    <th key={v} style={{
                      padding: "8px 6px",
                      background: "var(--card-bg, #16213e)",
                      borderBottom: "2px solid var(--border-color, #2a2a4a)",
                      textAlign: "center",
                      fontFamily: "monospace",
                      fontSize: 11,
                      fontWeight: 600,
                      whiteSpace: "nowrap",
                    }}>
                      {varNames[v]}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {filtered.map(rule => (
                  <tr key={rule.id}>
                    <td style={{
                      position: "sticky", left: 0, zIndex: 1,
                      padding: "6px 12px",
                      background: "var(--bg-color, #1a1a2e)",
                      borderBottom: "1px solid var(--border-color, #2a2a4a)",
                      fontFamily: "monospace",
                      fontWeight: 600,
                      opacity: 0.5,
                    }}>
                      R{rule.id}
                    </td>
                    {allActiveVars.map(v => {
                      const clause = rule.clauses.find(c => c.var === v);
                      return (
                        <td key={v} style={{
                          padding: "4px 6px",
                          borderBottom: "1px solid var(--border-color, #2a2a4a)",
                          textAlign: "center",
                        }}>
                          {clause ? (
                            <TermBadge termIndex={clause.term} termName={clause.termName} total={numTerms} />
                          ) : (
                            <span style={{ opacity: 0.15 }}>·</span>
                          )}
                        </td>
                      );
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}

        {filtered.length === 0 && (
          <p style={{ textAlign: "center", opacity: 0.4, marginTop: 40 }}>
            No rules match the current filter.
          </p>
        )}
      </div>
    </div>
  );
}