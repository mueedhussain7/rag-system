# app/evaluation/dashboard.py
import json
import sys
from pathlib import Path

# Make sure app/ is importable from the dashboard script
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from app.evaluation.logger import get_summary_stats, get_all_queries, get_all_ingestions

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RAG System Dashboard",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global styles ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Base */
html, body, [class*="css"] {
    font-family: 'Inter', 'Segoe UI', sans-serif;
}
.stApp {
    background: #080c18;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #0d1224;
    border-right: 1px solid #1e2d5a;
}
[data-testid="stSidebar"] .stRadio label {
    color: #a0aec0;
    font-size: 14px;
    padding: 6px 0;
}
[data-testid="stSidebar"] .stRadio label:hover {
    color: #7c9dff;
}

/* Metric cards */
.metric-card {
    background: linear-gradient(135deg, #0d1530 0%, #111a38 100%);
    border: 1px solid #1e3a6e;
    border-radius: 12px;
    padding: 20px 24px;
    position: relative;
    overflow: hidden;
}
.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, #3b6bff, #7c3aed);
}
.metric-label {
    color: #6b80a8;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 1.2px;
    text-transform: uppercase;
    margin-bottom: 8px;
}
.metric-value {
    color: #e2e8f0;
    font-size: 32px;
    font-weight: 700;
    line-height: 1;
    margin-bottom: 4px;
}
.metric-sub {
    color: #4a6080;
    font-size: 12px;
    margin-top: 6px;
}

/* Section headers */
.section-header {
    color: #7c9dff;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin: 32px 0 16px 0;
    padding-bottom: 8px;
    border-bottom: 1px solid #1e2d5a;
}

/* Confidence badges */
.badge-high     { background:#0d2e1a; color:#34d399; border:1px solid #065f46;
                  padding:3px 10px; border-radius:20px; font-size:12px; font-weight:600; }
.badge-medium   { background:#2d1f00; color:#fbbf24; border:1px solid #78350f;
                  padding:3px 10px; border-radius:20px; font-size:12px; font-weight:600; }
.badge-low      { background:#2d0f0f; color:#f87171; border:1px solid #7f1d1d;
                  padding:3px 10px; border-radius:20px; font-size:12px; font-weight:600; }
.badge-unverified { background:#1a1a2e; color:#94a3b8; border:1px solid #334155;
                    padding:3px 10px; border-radius:20px; font-size:12px; font-weight:600; }

/* Chart backgrounds */
.js-plotly-plot .plotly .bg { fill: transparent !important; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar navigation ────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding: 8px 0 24px 0;'>
        <div style='font-size:22px; font-weight:700; color:#7c9dff;'>⬡ RAG System</div>
        <div style='font-size:11px; color:#4a6080; letter-spacing:1px;
                    text-transform:uppercase; margin-top:4px;'>
            Evaluation Dashboard
        </div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        ["Overview", "Hallucination Tracker", "Query Explorer", "Document Manager"],
        label_visibility="collapsed",
    )

    st.markdown("<div style='height:32px'></div>", unsafe_allow_html=True)
    if st.button("↺  Refresh data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

# ── Shared chart theme ────────────────────────────────────────────────────────
CHART_THEME = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(13,18,36,0.6)",
    font=dict(color="#a0aec0", family="Inter, sans-serif", size=12),
    xaxis=dict(gridcolor="#1a2540", linecolor="#1e2d5a", tickcolor="#1e2d5a"),
    yaxis=dict(gridcolor="#1a2540", linecolor="#1e2d5a", tickcolor="#1e2d5a"),
    margin=dict(l=40, r=20, t=30, b=40),
)

COLORS = {
    "blue":   "#3b6bff",
    "purple": "#7c3aed",
    "teal":   "#0d9488",
    "green":  "#34d399",
    "amber":  "#fbbf24",
    "red":    "#f87171",
    "gray":   "#4a6080",
}

CONFIDENCE_COLORS = {
    "high":       COLORS["green"],
    "medium":     COLORS["amber"],
    "low":        COLORS["red"],
    "unverified": COLORS["gray"],
}

# ── Data loader ───────────────────────────────────────────────────────────────
@st.cache_data(ttl=30)
def load_data():
    stats     = get_summary_stats()
    queries   = get_all_queries()
    ingestions = get_all_ingestions()

    df = pd.DataFrame(queries) if queries else pd.DataFrame()
    if not df.empty:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["sources"]   = df["sources"].apply(
            lambda x: json.loads(x) if isinstance(x, str) else x
        )
    return stats, df, ingestions

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "Overview":
    stats, df, _ = load_data()

    st.markdown("""
    <div style='margin-bottom:28px;'>
        <h1 style='color:#e2e8f0; font-size:26px; font-weight:700; margin:0;'>
            System Overview
        </h1>
        <p style='color:#4a6080; font-size:13px; margin:6px 0 0 0;'>
            Live metrics across all queries and evaluations
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Metric cards ──────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Total Queries</div>
            <div class="metric-value">{stats['total_queries']}</div>
            <div class="metric-sub">questions answered</div>
        </div>""", unsafe_allow_html=True)

    with c2:
        faith = stats['avg_faithfulness']
        color = "#34d399" if faith >= 0.8 else "#fbbf24" if faith >= 0.5 else "#f87171"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Avg Faithfulness</div>
            <div class="metric-value" style="color:{color}">{faith:.2f}</div>
            <div class="metric-sub">RAGAS score (0–1)</div>
        </div>""", unsafe_allow_html=True)

    with c3:
        hal  = stats['hallucination_rate']
        color = "#34d399" if hal < 10 else "#fbbf24" if hal < 30 else "#f87171"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Hallucination Rate</div>
            <div class="metric-value" style="color:{color}">{hal}%</div>
            <div class="metric-sub">{stats['hallucination_count']} low-confidence answers</div>
        </div>""", unsafe_allow_html=True)

    with c4:
        lat = stats['avg_latency_ms']
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Avg Latency</div>
            <div class="metric-value">{lat:,.0f}<span style="font-size:16px;color:#4a6080"> ms</span></div>
            <div class="metric-sub">end-to-end per query</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)

    if df.empty:
        st.info("No queries logged yet — ask some questions via POST /ask to populate the dashboard.")
        st.stop()

    # ── Charts row ────────────────────────────────────────────────────────────
    left, right = st.columns([3, 2])

    with left:
        st.markdown('<div class="section-header">Faithfulness over time</div>',
                    unsafe_allow_html=True)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df["timestamp"], y=df["faithfulness_score"],
            mode="lines+markers",
            line=dict(color=COLORS["blue"], width=2),
            marker=dict(color=COLORS["blue"], size=6,
                        line=dict(color="#0d1224", width=1.5)),
            fill="tozeroy",
            fillcolor="rgba(59,107,255,0.08)",
            name="Faithfulness",
            hovertemplate="<b>%{y:.2f}</b><br>%{x|%H:%M %d %b}<extra></extra>",
        ))
        fig.add_hline(y=0.8, line_dash="dash",
                      line_color=COLORS["green"], opacity=0.5,
                      annotation_text="good threshold",
                      annotation_font_color=COLORS["green"])
        fig.update_layout(**CHART_THEME, height=260, showlegend=False)
        fig.update_yaxes(range=[0, 1.05])
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.markdown('<div class="section-header">Confidence distribution</div>',
                    unsafe_allow_html=True)
        counts = stats["confidence_counts"]
        labels = list(counts.keys())
        values = list(counts.values())
        colors = [CONFIDENCE_COLORS.get(l, COLORS["gray"]) for l in labels]

        fig2 = go.Figure(go.Pie(
            labels=labels, values=values,
            marker=dict(colors=colors,
                        line=dict(color="#080c18", width=2)),
            hole=0.6,
            textinfo="label+percent",
            textfont=dict(color="#a0aec0", size=12),
            hovertemplate="<b>%{label}</b><br>%{value} queries<extra></extra>",
        ))
        fig2.update_layout(**CHART_THEME, height=260, showlegend=False,
                           annotations=[dict(
                               text=f"{stats['total_queries']}<br><span style='font-size:10'>total</span>",
                               x=0.5, y=0.5, showarrow=False,
                               font=dict(color="#e2e8f0", size=18),
                           )])
        st.plotly_chart(fig2, use_container_width=True)

    # ── Latency distribution ──────────────────────────────────────────────────
    st.markdown('<div class="section-header">Latency distribution (ms)</div>',
                unsafe_allow_html=True)
    fig3 = go.Figure(go.Histogram(
        x=df["latency_ms"],
        nbinsx=20,
        marker=dict(
            color=COLORS["purple"],
            opacity=0.8,
            line=dict(color="#080c18", width=1),
        ),
        hovertemplate="<b>%{x:.0f} ms</b><br>%{y} queries<extra></extra>",
    ))
    fig3.update_layout(**CHART_THEME, height=200,
                       xaxis_title="Latency (ms)", yaxis_title="Queries",
                       bargap=0.1)
    st.plotly_chart(fig3, use_container_width=True)

    # ── Recent queries table ──────────────────────────────────────────────────
    st.markdown('<div class="section-header">Recent queries</div>',
                unsafe_allow_html=True)
    recent = df.head(5)[["timestamp", "question", "faithfulness_score",
                          "confidence_level", "latency_ms"]].copy()
    recent["timestamp"] = recent["timestamp"].dt.strftime("%d %b %H:%M")
    recent["question"]  = recent["question"].str[:70] + "…"
    recent.columns      = ["Time", "Question", "Faithfulness", "Confidence", "Latency (ms)"]
    st.dataframe(recent, use_container_width=True, hide_index=True)


# ═══════════════════════════
# PAGES 2-4 — PLACEHOLDERS
# ═══════════════════════════
# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — HALLUCINATION TRACKER
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Hallucination Tracker":
    _, df, _ = load_data()

    st.markdown("""
    <div style='margin-bottom:28px;'>
        <h1 style='color:#e2e8f0; font-size:26px; font-weight:700; margin:0;'>
            Hallucination Tracker
        </h1>
        <p style='color:#4a6080; font-size:13px; margin:6px 0 0 0;'>
            Faithfulness scores, confidence trends, and flagged answers
        </p>
    </div>
    """, unsafe_allow_html=True)

    if df.empty:
        st.info("No queries logged yet.")
        st.stop()

    valid = df[df["faithfulness_score"] >= 0].copy()

    # ── Score timeline with confidence bands ──────────────────────────────────
    st.markdown('<div class="section-header">Faithfulness score timeline</div>',
                unsafe_allow_html=True)

    fig = go.Figure()

    # Background bands showing danger / caution / good zones
    fig.add_hrect(y0=0, y1=0.5,
                  fillcolor=COLORS["red"], opacity=0.04, line_width=0,
                  annotation_text="danger zone",
                  annotation_font_color=COLORS["red"],
                  annotation_position="top left")
    fig.add_hrect(y0=0.5, y1=0.8,
                  fillcolor=COLORS["amber"], opacity=0.04, line_width=0,
                  annotation_text="caution",
                  annotation_font_color=COLORS["amber"],
                  annotation_position="top left")
    fig.add_hrect(y0=0.8, y1=1.0,
                  fillcolor=COLORS["green"], opacity=0.04, line_width=0,
                  annotation_text="good",
                  annotation_font_color=COLORS["green"],
                  annotation_position="top left")

    if not valid.empty:
        # Each point is coloured by its confidence level
        point_colors = [
            CONFIDENCE_COLORS.get(c, COLORS["gray"])
            for c in valid["confidence_level"]
        ]
        fig.add_trace(go.Scatter(
            x=valid["timestamp"],
            y=valid["faithfulness_score"],
            mode="lines+markers",
            line=dict(color=COLORS["blue"], width=1.5, dash="dot"),
            marker=dict(color=point_colors, size=10,
                        line=dict(color="#0d1224", width=1.5)),
            hovertemplate=(
                "<b>Score: %{y:.2f}</b><br>"
                "%{x|%H:%M %d %b}<extra></extra>"
            ),
        ))

    fig.update_layout(**CHART_THEME, height=300, showlegend=False)
    fig.update_yaxes(range=[0, 1.05])
    st.plotly_chart(fig, use_container_width=True)

    # ── Confidence breakdown + NLI verdict ───────────────────────────────────
    left, right = st.columns(2)

    with left:
        st.markdown('<div class="section-header">Confidence breakdown</div>',
                    unsafe_allow_html=True)
        conf_counts = df["confidence_level"].value_counts()
        fig2 = go.Figure()
        for level, color in CONFIDENCE_COLORS.items():
            count = conf_counts.get(level, 0)
            fig2.add_trace(go.Bar(
                name=level.capitalize(),
                x=[level.capitalize()],
                y=[count],
                marker_color=color,
                marker_line=dict(color="#080c18", width=1),
                hovertemplate=f"<b>{level}</b><br>%{{y}} queries<extra></extra>",
            ))
        fig2.update_layout(**CHART_THEME, height=260,
                           barmode="group", showlegend=True,
                           legend=dict(font=dict(color="#a0aec0")))
        st.plotly_chart(fig2, use_container_width=True)

    with right:
        st.markdown('<div class="section-header">NLI verdict distribution</div>',
                    unsafe_allow_html=True)
        nli_counts = df["nli_verdict"].value_counts()
        nli_colors = {
            "clean":        COLORS["green"],
            "uncertain":    COLORS["amber"],
            "contradicted": COLORS["red"],
        }
        fig3 = go.Figure(go.Pie(
            labels=nli_counts.index.tolist(),
            values=nli_counts.values.tolist(),
            marker=dict(
                colors=[nli_colors.get(l, COLORS["gray"])
                        for l in nli_counts.index],
                line=dict(color="#080c18", width=2),
            ),
            hole=0.55,
            textinfo="label+percent",
            textfont=dict(color="#a0aec0", size=12),
            hovertemplate="<b>%{label}</b><br>%{value} queries<extra></extra>",
        ))
        fig3.update_layout(**CHART_THEME, height=260, showlegend=False)
        st.plotly_chart(fig3, use_container_width=True)

    # ── Worst performing answers ──────────────────────────────────────────────
    st.markdown('<div class="section-header">Lowest faithfulness answers</div>',
                unsafe_allow_html=True)

    if valid.empty:
        st.info("No scored answers yet — ask questions via POST /ask first.")
    else:
        worst = (valid.nsmallest(5, "faithfulness_score")
                 [["timestamp","question","answer",
                   "faithfulness_score","confidence_level"]]
                 .copy())

        for _, row in worst.iterrows():
            badge_class = f"badge-{row['confidence_level']}"
            score_color = (
                COLORS["green"] if row["faithfulness_score"] >= 0.8
                else COLORS["amber"] if row["faithfulness_score"] >= 0.5
                else COLORS["red"]
            )
            st.markdown(f"""
            <div style="background:#0d1224; border:1px solid #1e2d5a;
                        border-radius:10px; padding:16px; margin-bottom:10px;">
                <div style="display:flex; justify-content:space-between;
                            align-items:center; margin-bottom:8px;">
                    <span style="color:#4a6080; font-size:12px;">
                        {row['timestamp'].strftime('%d %b %Y %H:%M')}
                    </span>
                    <div style="display:flex; gap:8px; align-items:center;">
                        <span style="color:{score_color}; font-weight:700; font-size:14px;">
                            {row['faithfulness_score']:.2f}
                        </span>
                        <span class="{badge_class}">{row['confidence_level']}</span>
                    </div>
                </div>
                <div style="color:#e2e8f0; font-size:13px; font-weight:500;
                            margin-bottom:6px;">
                    Q: {row['question'][:100]}
                </div>
                <div style="color:#a0aec0; font-size:12px; line-height:1.5;">
                    {row['answer'][:200]}…
                </div>
            </div>
            """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGES 3-4 — COMING NEXT
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Query Explorer":
    st.markdown("### 🚧 Coming next — Query Explorer")

elif page == "Document Manager":
    st.markdown("### 🚧 Coming next — Document Manager")
elif page == "Query Explorer":
    st.markdown("### Query Explorer")

elif page == "Document Manager":
    st.markdown("### Document Manager")