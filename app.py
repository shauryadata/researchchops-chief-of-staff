import json
import os
import re
import io

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 1) CSV Template Generator
# ============================================================
def build_template_csv_bytes() -> bytes:
    """
    Generic experiment-results CSV template (paper-agnostic).
    Required: method, metric_primary, metric_secondary
    Optional: secondary_constraint
    """
    template = pd.DataFrame(
        [
            {"method": "Baseline", "metric_primary": 0.82, "metric_secondary": 180.0, "secondary_constraint": 200.0},
            {"method": "Ablation", "metric_primary": 0.84, "metric_secondary": 190.0, "secondary_constraint": 200.0},
            {"method": "Proposed", "metric_primary": 0.89, "metric_secondary": 175.0, "secondary_constraint": 200.0},
        ]
    )
    return template.to_csv(index=False).encode("utf-8")


# ============================================================
# 2) Validation Logic (Strict)
# ============================================================
def validate_results_df(df: pd.DataFrame) -> tuple[bool, str]:
    required = {"method", "metric_primary", "metric_secondary"}
    missing = required - set(df.columns)
    if missing:
        return False, f"Missing required columns: {', '.join(sorted(missing))}"

    if df["method"].isna().any():
        return False, "Column 'method' contains empty values."

    for col in ["metric_primary", "metric_secondary"]:
        numeric_vals = pd.to_numeric(df[col], errors="coerce")
        if numeric_vals.isna().any():
            return False, f"Column '{col}' must contain only numeric values (no blanks or text)."

    # Optional constraint is allowed to be blank; if present, must be numeric where non-blank
    if "secondary_constraint" in df.columns:
        c_vals = pd.to_numeric(df["secondary_constraint"], errors="coerce")
        # if there are non-empty strings that become NaN, it will show as NaN; we allow blanks only.
        # We treat NaNs ok as long as at least one row has a number or all are blank.
        # No strict fail here; chart will ignore if none exist.

    return True, ""


# ============================================================
# 3) Charts
# ============================================================
def make_primary_metric_chart(df: pd.DataFrame):
    fig, ax = plt.subplots()
    d = df.copy()
    d["metric_primary"] = pd.to_numeric(d["metric_primary"], errors="coerce")
    ax.bar(d["method"].astype(str), d["metric_primary"])
    ax.set_xlabel("Method")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    return fig


def make_secondary_metric_chart(df: pd.DataFrame):
    fig, ax = plt.subplots()
    d = df.copy()
    d["metric_secondary"] = pd.to_numeric(d["metric_secondary"], errors="coerce")

    x = list(range(len(d)))
    ax.plot(x, d["metric_secondary"], marker="o")
    ax.set_xticks(x)
    ax.set_xticklabels(d["method"].astype(str), rotation=25)
    ax.set_xlabel("Method")

    # Optional constraint line
    if "secondary_constraint" in d.columns:
        constraint_vals = pd.to_numeric(d["secondary_constraint"], errors="coerce")
        if constraint_vals.notna().any():
            c = float(constraint_vals.dropna().median())
            ax.axhline(c, linestyle="--")
            ax.text(0, c, f" constraint={c:.3g}", va="bottom")

    fig.tight_layout()
    return fig


def fig_to_png_bytes(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200)
    buf.seek(0)
    plt.close(fig)  # prevent memory buildup
    return buf.read()


# ============================================================
# 4) Charts Helpers: baseline/proposed detection + captions
# ============================================================
def detect_rows(df: pd.DataFrame):
    """Detect baseline + proposed rows using common aliases."""
    m = df["method"].astype(str).str.strip().str.lower()

    baseline_aliases = {
        "baseline", "control", "default", "standard",
        "fcfs", "greedy", "naive", "heuristic", "no-shift", "no shift"
    }
    proposed_aliases = {
        "proposed", "ours", "our method", "new", "method a", "model a"
    }

    baseline_row = df[m.isin(baseline_aliases)]
    proposed_row = df[m.isin(proposed_aliases)]
    return baseline_row, proposed_row


def build_generic_caption(
    primary_name: str,
    secondary_name: str,
    baseline_label: str,
    proposed_label: str,
    base_p: float,
    prop_p: float,
    base_s: float,
    prop_s: float,
    secondary_constraint: float | None,
    primary_higher_is_better: bool,
):
    # Primary delta
    if base_p != 0:
        delta_pct = (prop_p - base_p) / abs(base_p) * 100.0
    else:
        delta_pct = None

    if delta_pct is None:
        primary_phrase = f"{primary_name} changes from {base_p:.3g} to {prop_p:.3g}."
    else:
        if primary_higher_is_better:
            primary_phrase = f"{primary_name} improves by {delta_pct:+.1f}% vs {baseline_label}."
        else:
            if delta_pct < 0:
                primary_phrase = f"{primary_name} reduces by {abs(delta_pct):.1f}% vs {baseline_label}."
            else:
                primary_phrase = f"{primary_name} increases by {delta_pct:.1f}% vs {baseline_label}."

    secondary_phrase = f"{secondary_name} changes from {base_s:.3g} to {prop_s:.3g}."
    if secondary_constraint is not None:
        secondary_phrase += f" The dashed line indicates a constraint of {secondary_constraint:.3g}."

    cap1 = f"Figure 1: {primary_name} by method. Compared to {baseline_label}, {proposed_label} {primary_phrase}"
    cap2 = f"Figure 2: {secondary_name} by method. Compared to {baseline_label}, {proposed_label} {secondary_phrase}"
    return cap1, cap2


# ============================================================
# Load env + client
# ============================================================
load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY", "")
client = OpenAI(api_key=API_KEY) if API_KEY else None


# ============================================================
# Page config
# ============================================================
st.set_page_config(
    page_title="ResearchOps Chief of Staff",
    page_icon="🧠",
    layout="wide",
)

st.markdown(
    """
    <style>
    /* Sticky header/tabs container */
    div[data-testid="stTabs"] {
        position: sticky;
        top: 0;
        z-index: 999;
        background: #0e1117; /* match Streamlit dark bg */
        padding-top: 0.5rem;
        border-bottom: 1px solid rgba(255,255,255,0.08);
    }

    /* Make the tab row look clean when sticky */
    div[data-testid="stTabs"] button {
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# State
# ============================================================
def init_state():
    defaults = {
        "hours_left": 20.0,
        "paper_text": "",
        "rubric_text": "",
        "reviewer_text": "",
        "project_brief": None,
        "deliverables": None,
        "last_error": "",
        "last_raw_llm": "",
        "last_raw_llm_repaired": "",

        # Demo mode
        "demo_mode": True,
        "demo_loaded": False,

        # Speaker scripts
        "speaker_scripts": None,
        "speaker_script_md": "",

        # Chart labels (persist across reruns)
        "primary_metric_label": "Accuracy",
        "secondary_metric_label": "Latency (ms)",
        "primary_higher_is_better": True,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def load_sample_demo():
    st.session_state.paper_text = (
        "Title: Carbon-Aware AI Scheduling for Hospital Microgrids\n\n"
        "Abstract:\n"
        "Hospitals increasingly rely on energy-intensive AI workloads for patient monitoring and clinical decision support. "
        "We propose a scheduling framework that shifts non-critical workloads while preserving response-time constraints. "
        "Using synthetic demand signals and real-world grid signals, we evaluate forecasting + shifting. Results indicate "
        "meaningful improvements while maintaining service constraints.\n"
    )
    st.session_state.rubric_text = (
        "Conference Rubric / CFP:\n"
        "1. Novelty and significance\n"
        "2. Technical correctness\n"
        "3. Experimental validation (baselines + ablations)\n"
        "4. Clarity of presentation\n"
        "5. Real-world impact and feasibility\n"
        "10. Reproducibility (optional example)\n"
    )
    st.session_state.reviewer_text = (
        "Reviewer Comments:\n"
        "- Add a clearer baseline comparison.\n"
        "- Add more detail on synthetic dataset generation.\n"
        "- Include ablations and sensitivity.\n"
        "- Clarify deployment assumptions.\n"
        "- Improve slide narrative.\n"
    )


def pretty_json(obj) -> str:
    return json.dumps(obj, indent=2, ensure_ascii=False)


# ============================================================
# Utility: rubric + hours
# ============================================================
def extract_rubric_criteria(rubric_text: str):
    lines = [ln.strip() for ln in rubric_text.splitlines() if ln.strip()]
    criteria = []
    num_pat = re.compile(r"^\s*(\d+)\s*[\.\)]\s*(.+)\s*$")

    for ln in lines:
        m = num_pat.match(ln)
        if m:
            criteria.append(m.group(2).strip())
            continue
        if ln.startswith(("-", "•")):
            criteria.append(ln[1:].strip())

    if not criteria and len(lines) > 1:
        for ln in lines[1:]:
            if len(ln) > 4:
                criteria.append(ln)

    seen = set()
    out = []
    for c in criteria:
        if c not in seen:
            out.append(c)
            seen.add(c)
    return out


def compute_plan_hours(deliverables: dict):
    total = 0.0
    for t in (deliverables.get("plan", []) or []):
        try:
            total += float(t.get("eta_hours", 0) or 0)
        except Exception:
            pass
    return total


# ============================================================
# LLM helpers (STRICT JSON MODE)
# ============================================================
def call_llm_json(prompt: str) -> dict:
    if client is None:
        raise RuntimeError("OPENAI_API_KEY not found. Add it to .env and restart Streamlit.")

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Return ONLY valid JSON. No markdown. No commentary."},
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_object"},
        temperature=0.2,
        max_tokens=2500,
    )

    raw = (resp.choices[0].message.content or "").strip()
    st.session_state.last_raw_llm = raw
    st.session_state.last_raw_llm_repaired = ""
    return json.loads(raw)


def build_project_brief_prompt(paper: str, rubric: str, reviewer: str) -> str:
    return f"""
Create a project_brief JSON from the following inputs.

PAPER:
\"\"\"{paper}\"\"\"

RUBRIC/CFP:
\"\"\"{rubric}\"\"\"

REVIEWER COMMENTS (optional):
\"\"\"{reviewer}\"\"\"

Return ONLY valid JSON with exactly these keys:
title, domain, one_liner, problem, proposed_solution,
data, methods, metrics, key_claims, current_gaps, target_venue_fit

Where:
- data is an object like {{ "sources": [], "synthetic": false, "notes": "" }}
- target_venue_fit is an object like {{ "venue": "", "rubric_points": [] }}
"""


def build_deliverables_prompt(project_brief: dict, hours_left: float, rubric_criteria: list[str]) -> str:
    """
    - Forces rubric_alignment criteria to match rubric items EXACTLY.
    - Forces required plan task for charts (paper-agnostic).
    """
    rubric_list_json = json.dumps(rubric_criteria, ensure_ascii=False, indent=2)

    return f"""
You must output VALID JSON ONLY.

Return a JSON object with EXACTLY these keys:
plan, experiments, risks, slides, rubric_alignment

Each key maps to an ARRAY of objects with EXACT field names:

plan: [
  {{
    "task": "...",
    "priority": "High|Medium|Low",
    "eta_hours": 0.0,
    "acceptance_criteria": "...",
    "audit": {{"evidence_quote": "...", "assumption": "...", "confidence": 0.0}}
  }}
]

experiments: [
  {{
    "experiment": "...",
    "baseline_method": "...",
    "metric": "...",
    "chart": "...",
    "why": "...",
    "expected_impact": "...",
    "audit": {{"evidence_quote": "...", "assumption": "...", "confidence": 0.0}}
  }}
]

risks: [
  {{
    "risk": "...",
    "severity": "High|Medium|Low",
    "mitigation": "...",
    "audit": {{"evidence_quote": "...", "assumption": "...", "confidence": 0.0}}
  }}
]

slides: [
  {{
    "slide_title": "...",
    "bullets": ["...", "..."],
    "speaker_notes": "...",
    "audit": {{"evidence_quote": "...", "assumption": "...", "confidence": 0.0}}
  }}
]

rubric_alignment: [
  {{
    "criterion": "...",
    "status": "yes|partial|no",
    "fix": "...",
    "audit": {{"evidence_quote": "...", "assumption": "...", "confidence": 0.0}}
  }}
]

Constraints:
- Total eta_hours across ALL plan items must be <= {hours_left}
- plan MUST contain at least 5 items total (including the required charts task)
- slides must have 8 to 10 items
- experiments must contain at least 3 items, including:
  1) baseline comparison
  2) ablation study
  3) sensitivity analysis

Experiment requirements:
- Each experiment object MUST include these non-empty fields:
  • baseline_method
  • metric
  • chart
- baseline_method must be a specific named baseline (e.g., "No-shift scheduling", "FCFS", "Greedy scheduler")
- metric must be explicit and measurable (e.g., "Accuracy", "Average latency (ms)", "p95 latency")
- chart must specify the exact visualization type (e.g., "bar chart", "line plot", "scatter plot")

Slides MUST include:
- one slide titled exactly: "System Architecture"
- one slide titled exactly: "Results (Charts)"

Plan MUST include a task exactly named:
"Build 2 charts (primary metric vs baseline, secondary metric vs constraint)"
- That task must have eta_hours between 1.0 and 2.0

Rubric alignment MUST satisfy:
- criterion MUST be EXACTLY one of the following rubric criteria strings
- rubric_alignment MUST contain EVERY rubric criterion exactly once

Rubric criteria list (must match exactly):
{rubric_list_json}

Project brief JSON:
{json.dumps(project_brief, indent=2, ensure_ascii=False)}

Now produce deliverables JSON.
"""


def normalize_deliverables(d: dict) -> dict:
    if not isinstance(d, dict):
        return {"plan": [], "experiments": [], "risks": [], "slides": [], "rubric_alignment": []}
    for k in ["plan", "experiments", "risks", "slides", "rubric_alignment"]:
        if k not in d or d[k] is None:
            d[k] = []
    return d


def generate_all():
    st.session_state.last_error = ""

    paper = st.session_state.paper_text.strip()
    rubric = st.session_state.rubric_text.strip()
    reviewer = st.session_state.reviewer_text.strip()

    if not paper or not rubric:
        st.session_state.last_error = "Please provide at least Paper + Rubric."
        return

    rubric_criteria = extract_rubric_criteria(rubric)

    with st.spinner("1/2 Generating project brief..."):
        brief_prompt = build_project_brief_prompt(paper, rubric, reviewer)
        st.session_state.project_brief = call_llm_json(brief_prompt)

    with st.spinner("2/2 Generating deliverables..."):
        deliver_prompt = build_deliverables_prompt(
            st.session_state.project_brief,
            float(st.session_state.hours_left),
            rubric_criteria,
        )
        st.session_state.deliverables = normalize_deliverables(call_llm_json(deliver_prompt))


# ============================================================
# Speaker Script Generator (D)
# ============================================================
def build_speaker_script_prompt(project_brief: dict, slides: list[dict]) -> str:
    return f"""
You must output VALID JSON ONLY.

Create speaker scripts for the following slide outline.
Target length per slide: 60–110 words (≈30–45 seconds).
Tone: confident, clear, product-demo friendly (not academic jargon-heavy).
Do NOT mention CO₂ unless the slide explicitly mentions it.
If a slide includes charts/results, describe what the viewer should notice.

Return a JSON object with EXACTLY these keys:
scripts, opener, closer, qa_bank

Where:
scripts: [
  {{
    "slide_title": "...",
    "script": "...",
    "key_points": ["...", "..."],
    "transition_to_next": "..."
  }}
]
opener: "..."  (15–25 sec opening for the entire talk)
closer: "..."  (15–25 sec closing)
qa_bank: [
  {{ "question": "...", "answer": "..." }}
]

Project brief:
{json.dumps(project_brief, ensure_ascii=False, indent=2)}

Slides:
{json.dumps(slides, ensure_ascii=False, indent=2)}
"""


def generate_speaker_scripts():
    st.session_state.last_error = ""

    if not st.session_state.deliverables:
        st.session_state.last_error = "Generate deliverables first (Inputs → Generate)."
        return

    pb = st.session_state.project_brief or {}
    slides = (st.session_state.deliverables or {}).get("slides", []) or []
    if not slides:
        st.session_state.last_error = "No slides found. Generate deliverables again."
        return

    prompt = build_speaker_script_prompt(pb, slides)
    with st.spinner("Generating speaker scripts..."):
        scripts_json = call_llm_json(prompt)

    st.session_state.speaker_scripts = scripts_json

    # Markdown export
    md = "# Speaker Script\n\n"
    md += f"## Opener\n{(scripts_json.get('opener','') or '').strip()}\n\n"
    for s in scripts_json.get("scripts", []) or []:
        title = (s.get("slide_title") or "").strip()
        script = (s.get("script") or "").strip()
        md += f"## {title}\n{script}\n\n"
        kp = s.get("key_points", []) or []
        if kp:
            md += "**Key points:**\n"
            for p in kp:
                md += f"- {p}\n"
            md += "\n"
        trans = (s.get("transition_to_next") or "").strip()
        if trans:
            md += f"**Transition:** {trans}\n\n"
    md += f"## Closer\n{(scripts_json.get('closer','') or '').strip()}\n\n"

    qa = scripts_json.get("qa_bank", []) or []
    if qa:
        md += "## Q&A Bank\n"
        for item in qa:
            q = (item.get("question") or "").strip()
            a = (item.get("answer") or "").strip()
            md += f"- **Q:** {q}\n  **A:** {a}\n\n"

    st.session_state.speaker_script_md = md


# ============================================================
# Render helpers
# ============================================================
def render_list_section(title: str, section_key: str, primary_field: str):
    st.subheader(title)
    data = st.session_state.deliverables
    if not data:
        st.warning("Not generated yet. Go to Inputs → click **Generate**.", icon="⚠️")
        return

    items = data.get(section_key, []) or []
    if not items:
        st.info("No items generated for this section.")
        return

    for i, item in enumerate(items, start=1):
        heading = (item or {}).get(primary_field, f"Item {i}")
        st.markdown(f"### {heading}")

        if "priority" in item or "eta_hours" in item:
            meta = []
            if "priority" in item:
                meta.append(f"**Priority:** {item.get('priority','')}")
            if "eta_hours" in item:
                meta.append(f"**ETA (hrs):** {item.get('eta_hours','')}")
            st.write(" • ".join(meta))

        if section_key == "slides":
            bullets = item.get("bullets", []) or []
            if bullets:
                st.write("**Bullets:**")
                for b in bullets:
                    st.write(f"- {b}")
            notes = item.get("speaker_notes", "")
            if notes:
                st.write("**Speaker notes:**")
                st.write(notes)

        if section_key == "rubric_alignment":
            st.write(f"**Status:** {item.get('status','')}")
            st.write(f"**Fix:** {item.get('fix','')}")

        audit = item.get("audit", {}) or {}
        with st.expander("Audit trail"):
            st.write(f"**Evidence quote:** {audit.get('evidence_quote','')}")
            st.write(f"**Assumption:** {audit.get('assumption','')}")
            st.write(f"**Confidence:** {audit.get('confidence','')}")

        st.divider()


# ============================================================
# Init app
# ============================================================
init_state()

# ============================================================
# Sidebar (Polished + Demo mode)
# ============================================================
st.sidebar.title("🧠 ResearchOps Chief of Staff")
st.sidebar.caption("Turn research chaos into a submission-ready plan.")
st.sidebar.divider()

st.sidebar.subheader("Demo Mode")
st.session_state.demo_mode = st.sidebar.toggle(
    "Enable demo mode (judge-friendly UI)",
    value=bool(st.session_state.demo_mode),
)

if st.session_state.demo_mode and not st.session_state.demo_loaded:
    load_sample_demo()
    st.session_state.demo_loaded = True

st.sidebar.caption("Demo mode hides debug blocks and keeps the UI product-like.")
st.sidebar.divider()

st.session_state.hours_left = st.sidebar.number_input(
    "Hours left (planning constraint)",
    min_value=1.0,
    max_value=72.0,
    value=float(st.session_state.hours_left),
    step=1.0,
)

st.sidebar.divider()
col_a, col_b = st.sidebar.columns(2)
with col_a:
    if st.sidebar.button("📦 Load Sample Demo", use_container_width=True):
        load_sample_demo()
        st.session_state.demo_loaded = True
with col_b:
    if st.sidebar.button("🔄 Reset", use_container_width=True):
        for key in ["paper_text", "rubric_text", "reviewer_text"]:
            st.session_state[key] = ""
        st.session_state.project_brief = None
        st.session_state.deliverables = None
        st.session_state.speaker_scripts = None
        st.session_state.speaker_script_md = ""
        st.session_state.last_error = ""
        st.session_state.last_raw_llm = ""
        st.session_state.last_raw_llm_repaired = ""
        st.session_state.demo_loaded = False

st.sidebar.divider()
st.sidebar.markdown(
    "**Build status:**\n"
    "- ✅ Step 1: UI skeleton\n"
    "- ✅ Step 2: LLM pipeline\n"
    "- ✅ Step 3: Checks + Export\n"
    "- ✅ Step 4: Charts (paper-agnostic)\n"
    "- ✅ Step 5: Speaker script generator\n"
)

# ============================================================
# Main header
# ============================================================
st.title("ResearchOps Chief of Staff")
st.caption("Paste paper + rubric + reviewer comments → generate plan, experiments, risks, slides, scorecard (with audit trail).")

if not API_KEY:
    st.error("OPENAI_API_KEY not found. Add it to your .env file and restart Streamlit.")

st.markdown(
    """
    <style>
    /* Streamlit app main scroll container */
    section[data-testid="stSidebar"] + section main {
        overflow: auto;
    }

    /* Make the Tabs bar sticky within Streamlit's scroll container */
    div[data-testid="stTabs"] {
        position: sticky;
        top: 0.75rem;      /* adjust if it overlaps */
        z-index: 9999;
        background: #0e1117;
        padding-top: 0.25rem;
        padding-bottom: 0.25rem;
        border-bottom: 1px solid rgba(255,255,255,0.08);
        box-shadow: 0 2px 6px rgba(0,0,0,0.35);
    }

    /* Reduce extra whitespace above tabs */
    div[data-testid="stTabs"] + div {
        margin-top: 0.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ✅ FINAL tab layout (fixed indices)
tabs = st.tabs([
    "Inputs",
    "Plan",
    "Experiments",
    "Risks",
    "Slides",
    "Rubric Scorecard",
    "Checks",
    "Charts",
    "Speaker Scripts",
    "Export",
])


# ============================================================
# Inputs tab
# ============================================================
with tabs[0]:
    st.subheader("Inputs")
    st.write("For judging, enable **Demo mode** in the sidebar (or click **Load Sample Demo**).")

    left, right = st.columns(2, gap="large")

    with left:
        st.session_state.paper_text = st.text_area(
            "Paper Draft / Abstract (paste text)",
            value=st.session_state.paper_text,
            height=260,
            placeholder="Paste your abstract or draft here…",
        )
        st.session_state.reviewer_text = st.text_area(
            "Reviewer Comments (optional)",
            value=st.session_state.reviewer_text,
            height=220,
            placeholder="Paste reviewer bullets here (optional)…",
        )

    with right:
        st.session_state.rubric_text = st.text_area(
            "Conference CFP / Rubric (paste text)",
            value=st.session_state.rubric_text,
            height=260,
            placeholder="Paste the CFP/rubric criteria here…",
        )
        st.info(
            "Click **Generate** to produce:\n"
            "- project brief\n"
            "- deliverables (plan, experiments, risks, slides, rubric scorecard)\n",
            icon="ℹ️",
        )

    if st.button("🚀 Generate Chief of Staff Outputs", use_container_width=True, disabled=(not API_KEY)):
        try:
            generate_all()
            if st.session_state.last_error:
                st.error(st.session_state.last_error)
            else:
                st.success("Generated! Open the other tabs to view results.")
        except Exception as e:
            st.session_state.last_error = str(e)
            st.error(f"Generation failed: {st.session_state.last_error}")

    # ✅ In demo mode, keep UI clean (no debug by default)
    if st.session_state.project_brief and (not st.session_state.demo_mode):
        with st.expander("View project_brief.json"):
            st.code(pretty_json(st.session_state.project_brief), language="json")

    if st.session_state.deliverables and (not st.session_state.demo_mode):
        with st.expander("View deliverables.json"):
            st.code(pretty_json(st.session_state.deliverables), language="json")

    if st.session_state.get("last_raw_llm") and (not st.session_state.demo_mode):
        with st.expander("Debug: last raw LLM output"):
            st.code(st.session_state["last_raw_llm"])


# ============================================================
# Output tabs
# ============================================================
with tabs[1]:
    render_list_section("Plan (Tasks + Timeline)", "plan", "task")

with tabs[2]:
    render_list_section("Experiments Backlog", "experiments", "experiment")

with tabs[3]:
    render_list_section("Risk Register", "risks", "risk")

with tabs[4]:
    render_list_section("Slide Outline", "slides", "slide_title")

with tabs[5]:
    render_list_section("Rubric Alignment Scorecard", "rubric_alignment", "criterion")


# ============================================================
# Checks tab
# ============================================================
with tabs[6]:
    st.subheader("Checks")
    if not st.session_state.deliverables:
        st.warning("Nothing to check yet. Generate outputs in Inputs.", icon="⚠️")
    else:
        d = st.session_state.deliverables
        hours_left = float(st.session_state.hours_left)

        total_hours = compute_plan_hours(d)
        slides_n = len(d.get("slides", []) or [])
        exp_n = len(d.get("experiments", []) or [])
        risks_n = len(d.get("risks", []) or [])

        rubric_src = st.session_state.rubric_text or ""
        rubric_criteria = extract_rubric_criteria(rubric_src)
        aligned = [x.get("criterion", "").strip() for x in (d.get("rubric_alignment", []) or [])]
        missing = [c for c in rubric_criteria if c not in aligned]

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Plan hours (sum ETA)", f"{total_hours:.1f}h", delta=f"{hours_left - total_hours:.1f}h left")
        c2.metric("# Slides", str(slides_n))
        c3.metric("# Experiments", str(exp_n))
        c4.metric("# Risks", str(risks_n))

        st.divider()

        checks = []
        checks.append(("Plan fits in time budget", total_hours <= hours_left,
                       f"Total plan ETA is {total_hours:.1f}h but budget is {hours_left:.1f}h."))
        checks.append(("Slides count is 8–10", 8 <= slides_n <= 10,
                       f"Slides count is {slides_n}. Target is 8–10."))
        checks.append(("At least 3 experiments (baseline + ablation + sensitivity)", exp_n >= 3,
                       f"Experiments count is {exp_n}. Target ≥3."))
        checks.append(("Rubric alignment covers all rubric criteria", len(missing) == 0,
                       f"Missing rubric criteria: {', '.join(missing) if missing else ''}"))

        for name, ok, fail_msg in checks:
            if ok:
                st.success(f"✅ {name}")
            else:
                st.error(f"❌ {name}\n\n{fail_msg}")

        st.divider()
        st.subheader("Recommended next improvements (fast)")
        suggestions = []
        if exp_n < 3:
            suggestions.append("- Add **baseline**: no-shift / greedy / FCFS vs your approach.")
            suggestions.append("- Add **ablation**: XGBoost vs naive; LSTM vs heuristic shifting.")
            suggestions.append("- Add **sensitivity**: vary workload size, constraint, operating conditions.")
        if missing:
            suggestions.append("- Ensure rubric_alignment uses rubric criteria **exactly as written**.")
        if total_hours > hours_left:
            suggestions.append("- Reduce plan ETA to fit time budget.")

        st.markdown("\n".join(suggestions) if suggestions else "Looks solid. Next: upload results CSV to generate charts + captions.")


# ============================================================
# Charts tab
# ============================================================
with tabs[7]:
    st.subheader("Charts (Upload Results CSV → Auto-Figures)")

    st.markdown(
        """
Upload a CSV containing your experiment results.  
This is **paper-agnostic**.

**Required columns:** `method`, `metric_primary`, `metric_secondary`  
**Optional column:** `secondary_constraint`
"""
    )

    cA, cB, cC = st.columns([1, 1, 1])
    with cA:
        st.session_state.primary_metric_label = st.text_input(
            "Primary metric label",
            value=st.session_state.primary_metric_label,
        )
    with cB:
        st.session_state.secondary_metric_label = st.text_input(
            "Secondary metric label",
            value=st.session_state.secondary_metric_label,
        )
    with cC:
        direction = st.selectbox(
            "Primary metric direction",
            options=["Higher is better", "Lower is better"],
            index=0 if st.session_state.primary_higher_is_better else 1,
        )
        st.session_state.primary_higher_is_better = (direction == "Higher is better")

    st.download_button(
        label="⬇️ Download CSV Template",
        data=build_template_csv_bytes(),
        file_name="results_template.csv",
        mime="text/csv",
        use_container_width=True,
    )

    uploaded = st.file_uploader("Upload results CSV", type=["csv"])

    if uploaded is None:
        st.info("Upload a CSV to generate charts + captions.", icon="ℹ️")
    else:
        try:
            df = pd.read_csv(uploaded)
            df.columns = [c.strip() for c in df.columns]

            ok, msg = validate_results_df(df)
            if not ok:
                st.error(msg)
            else:
                st.success("CSV looks valid. Generating charts…")

                primary_name = st.session_state.primary_metric_label
                secondary_name = st.session_state.secondary_metric_label
                higher_is_better = bool(st.session_state.primary_higher_is_better)

                fig_p = make_primary_metric_chart(df)
                fig_s = make_secondary_metric_chart(df)

                fig_p.axes[0].set_title(f"{primary_name} by Method")
                fig_p.axes[0].set_ylabel(primary_name)

                fig_s.axes[0].set_title(f"{secondary_name} by Method")
                fig_s.axes[0].set_ylabel(secondary_name)

                c1, c2 = st.columns(2)
                with c1:
                    st.pyplot(fig_p, clear_figure=True)
                with c2:
                    st.pyplot(fig_s, clear_figure=True)

                p_png = fig_to_png_bytes(fig_p)
                s_png = fig_to_png_bytes(fig_s)

                d1, d2 = st.columns(2)
                with d1:
                    st.download_button(
                        "⬇️ Download Primary Chart (PNG)",
                        data=p_png,
                        file_name="chart_primary.png",
                        mime="image/png",
                        use_container_width=True,
                    )
                with d2:
                    st.download_button(
                        "⬇️ Download Secondary Chart (PNG)",
                        data=s_png,
                        file_name="chart_secondary.png",
                        mime="image/png",
                        use_container_width=True,
                    )

                st.divider()
                st.subheader("Quick Summary + Captions")

                d = df.copy()
                d["metric_primary"] = pd.to_numeric(d["metric_primary"], errors="coerce")
                d["metric_secondary"] = pd.to_numeric(d["metric_secondary"], errors="coerce")

                baseline_row, proposed_row = detect_rows(d)

                if len(baseline_row) == 1 and len(proposed_row) == 1:
                    base_method = str(baseline_row["method"].iloc[0])
                    prop_method = str(proposed_row["method"].iloc[0])

                    base_p = float(baseline_row["metric_primary"].iloc[0])
                    prop_p = float(proposed_row["metric_primary"].iloc[0])
                    base_s = float(baseline_row["metric_secondary"].iloc[0])
                    prop_s = float(proposed_row["metric_secondary"].iloc[0])

                    constraint_val = None
                    if "secondary_constraint" in d.columns:
                        c_vals = pd.to_numeric(d["secondary_constraint"], errors="coerce").dropna()
                        if len(c_vals) > 0:
                            constraint_val = float(c_vals.median())

                    if base_p != 0:
                        delta_pct = (prop_p - base_p) / abs(base_p) * 100.0
                        if higher_is_better:
                            st.write(f"- **{primary_name} change vs {base_method}:** {delta_pct:+.1f}%")
                        else:
                            if delta_pct < 0:
                                st.write(f"- **{primary_name} reduction vs {base_method}:** {abs(delta_pct):.1f}%")
                            else:
                                st.write(f"- **{primary_name} increase vs {base_method}:** {delta_pct:.1f}%")

                    st.write(f"- **{secondary_name} ({base_method} → {prop_method}):** {base_s:.3g} → {prop_s:.3g}")
                    if constraint_val is not None:
                        st.write(f"- **Constraint (median):** {constraint_val:.3g}")

                    cap1, cap2 = build_generic_caption(
                        primary_name=primary_name,
                        secondary_name=secondary_name,
                        baseline_label=base_method,
                        proposed_label=prop_method,
                        base_p=base_p,
                        prop_p=prop_p,
                        base_s=base_s,
                        prop_s=prop_s,
                        secondary_constraint=constraint_val,
                        primary_higher_is_better=higher_is_better,
                    )

                    st.write("**Copy/paste captions:**")
                    st.code(cap1, language="text")
                    st.code(cap2, language="text")

                else:
                    st.info(
                        "Tip: For auto baseline comparison, include one row named "
                        "**Baseline/Control/FCFS/Greedy** and one named **Proposed/Ours/New**.",
                        icon="ℹ️",
                    )
        except Exception as e:
            st.error(f"Could not read CSV: {e}")


# ============================================================
# Speaker Scripts tab
# ============================================================
with tabs[8]:
    st.subheader("Speaker Script Generator")

    if not API_KEY:
        st.warning("Add OPENAI_API_KEY to use speaker scripts.", icon="⚠️")

    if not st.session_state.deliverables:
        st.info("Generate deliverables first (Inputs → Generate).", icon="ℹ️")
    else:
        c1, c2, c3 = st.columns([1, 1, 2])
        with c1:
            if st.button("🎤 Generate Speaker Scripts", use_container_width=True, disabled=(not API_KEY)):
                try:
                    generate_speaker_scripts()
                    if st.session_state.last_error:
                        st.error(st.session_state.last_error)
                    else:
                        st.success("Speaker scripts generated.")
                except Exception as e:
                    st.error(f"Speaker script generation failed: {e}")
        with c2:
            if st.button("🧹 Clear", use_container_width=True):
                st.session_state.speaker_scripts = None
                st.session_state.speaker_script_md = ""
        with c3:
            if st.session_state.speaker_script_md:
                st.download_button(
                    "⬇️ Download speaker_script.md",
                    data=st.session_state.speaker_script_md,
                    file_name="speaker_script.md",
                    mime="text/markdown",
                    use_container_width=True,
                )

        if st.session_state.speaker_scripts:
            scripts = st.session_state.speaker_scripts

            st.markdown("### Opener")
            st.write(scripts.get("opener", ""))

            st.divider()
            st.markdown("### Per-slide scripts")
            for s in scripts.get("scripts", []) or []:
                st.markdown(f"**{s.get('slide_title','')}**")
                st.write(s.get("script", ""))

                kp = s.get("key_points", []) or []
                if kp:
                    st.write("Key points:")
                    for p in kp:
                        st.write(f"- {p}")

                tr = s.get("transition_to_next", "")
                if tr:
                    st.write(f"Transition: {tr}")

                st.divider()

            st.markdown("### Closer")
            st.write(scripts.get("closer", ""))

            qa = scripts.get("qa_bank", []) or []
            if qa:
                st.markdown("### Q&A Bank")
                for item in qa:
                    st.write(f"**Q:** {item.get('question','')}")
                    st.write(f"**A:** {item.get('answer','')}")
                    st.divider()


# ============================================================
# Export tab
# ============================================================
with tabs[9]:
    st.subheader("Export")

    if not st.session_state.deliverables:
        st.warning("Nothing to export yet. Generate deliverables in Inputs.", icon="⚠️")
    else:
        md = "# ResearchOps Chief of Staff Report\n\n"
        pb = st.session_state.project_brief or {}

        md += f"## Title\n{pb.get('title','')}\n\n"
        md += f"## One-liner\n{pb.get('one_liner','')}\n\n"

        md += "## Plan\n"
        for t in st.session_state.deliverables.get("plan", []):
            md += f"- **{t.get('task','')}** ({t.get('priority','')}, {t.get('eta_hours','')}h) — {t.get('acceptance_criteria','')}\n"

        md += "\n## Experiments\n"
        for x in st.session_state.deliverables.get("experiments", []):
            md += f"- **{x.get('experiment','')}** — {x.get('why','')} (Impact: {x.get('expected_impact','')})\n"

        md += "\n## Risks\n"
        for r in st.session_state.deliverables.get("risks", []):
            md += f"- **{r.get('risk','')}** ({r.get('severity','')}) — Mitigation: {r.get('mitigation','')}\n"

        md += "\n## Slides\n"
        for s in st.session_state.deliverables.get("slides", []):
            md += f"- **{s.get('slide_title','')}**\n"

        md += "\n## Rubric Alignment\n"
        for a in st.session_state.deliverables.get("rubric_alignment", []):
            md += f"- **{a.get('criterion','')}**: {a.get('status','')} — Fix: {a.get('fix','')}\n"

        # If speaker scripts exist, include a link section in the report
        if st.session_state.speaker_script_md:
            md += "\n## Speaker Script\n(Downloaded separately as `speaker_script.md`)\n"

        st.download_button(
            label="⬇️ Download researchops_report.md",
            data=md,
            file_name="researchops_report.md",
            mime="text/markdown",
            use_container_width=True,
        )
        st.code(md, language="markdown")
