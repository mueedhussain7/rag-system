# app/evaluation/dashboard.py
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from app.evaluation.logger import get_summary_stats, get_all_queries, get_all_ingestions

#Page config
st.set_page_config(
    page_title="RAG System Dashboard",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Theme state
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = True

#Theme tokens
DARK = {
    "bg":            "#080c18",
    "surface":       "#0d1224",
    "surface2":      "#111a38",
    "border":        "#1e2d5a",
    "border2":       "#1e3a6e",
    "text":          "#e2e8f0",
    "text_muted":    "#a0aec0",
    "text_dim":      "#6b80a8",
    "text_dimmer":   "#4a6080",
    "accent":        "#3b6bff",
    "accent2":       "#7c3aed",
    "section":       "#7c9dff",
    "card_grad1":    "#0d1530",
    "card_grad2":    "#111a38",
    "card_top":      "linear-gradient(90deg, #3b6bff, #7c3aed)",
    "chart_bg":      "rgba(13,18,36,0.6)",
    "chart_grid":    "#1a2540",
    "chart_line":    "#1e2d5a",
    "chart_fill":    "rgba(59,107,255,0.08)",
    "pie_outline":   "#080c18",
    "green":         "#34d399",
    "amber":         "#fbbf24",
    "red":           "#f87171",
    "blue":          "#3b6bff",
    "purple":        "#7c3aed",
    "gray":          "#4a6080",
    "badge_high_bg": "#0d2e1a", "badge_high_fg": "#34d399", "badge_high_bd": "#065f46",
    "badge_med_bg":  "#2d1f00", "badge_med_fg":  "#fbbf24", "badge_med_bd":  "#78350f",
    "badge_low_bg":  "#2d0f0f", "badge_low_fg":  "#f87171", "badge_low_bd":  "#7f1d1d",
    "badge_unv_bg":  "#1a1a2e", "badge_unv_fg":  "#94a3b8", "badge_unv_bd":  "#334155",
    "pill_bg":       "#111a38",
    "pill_fg":       "#6b80a8",
    "sidebar_bg":    "#0d1224",
    "sidebar_hover": "rgba(124,157,255,0.08)",
    "qcard_bg":      "#0d1224",
    "qcard_bd":      "#1e2d5a",
    "table_header":  "#080c18",
    "table_row_alt": "#0a0f1e",
}

LIGHT = {
    "bg":            "#f7f8fc",
    "surface":       "#ffffff",
    "surface2":      "#f0f2f9",
    "border":        "#e4e7f0",
    "border2":       "#d0d5e8",
    "text":          "#1a1d2e",
    "text_muted":    "#4a5068",
    "text_dim":      "#7b82a0",
    "text_dimmer":   "#9da4bf",
    "accent":        "#6366f1",
    "accent2":       "#8b5cf6",
    "section":       "#6366f1",
    "card_grad1":    "#ffffff",
    "card_grad2":    "#f5f6ff",
    "card_top":      "linear-gradient(90deg, #6366f1, #8b5cf6)",
    "chart_bg":      "rgba(247,248,252,0.9)",
    "chart_grid":    "#eceef8",
    "chart_line":    "#e4e7f0",
    "chart_fill":    "rgba(99,102,241,0.05)",
    "pie_outline":   "#f7f8fc",
    "green":         "#059669",
    "amber":         "#d97706",
    "red":           "#dc2626",
    "blue":          "#6366f1",
    "purple":        "#8b5cf6",
    "gray":          "#9da4bf",
    "badge_high_bg": "#ecfdf5", "badge_high_fg": "#065f46", "badge_high_bd": "#a7f3d0",
    "badge_med_bg":  "#fffbeb", "badge_med_fg":  "#92400e", "badge_med_bd":  "#fcd34d",
    "badge_low_bg":  "#fef2f2", "badge_low_fg":  "#991b1b", "badge_low_bd":  "#fca5a5",
    "badge_unv_bg":  "#f8fafc", "badge_unv_fg":  "#475569", "badge_unv_bd":  "#cbd5e1",
    "pill_bg":       "#eef2ff",
    "pill_fg":       "#6366f1",
    "sidebar_bg":    "#ffffff",
    "sidebar_hover": "rgba(99,102,241,0.06)",
    "qcard_bg":      "#ffffff",
    "qcard_bd":      "#e4e7f0",
    "table_header":  "#f0f2f9",
    "table_row_alt": "#f7f8fc",
}

T = DARK if st.session_state.dark_mode else LIGHT

#CSS
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"], .stApp {{
    background-color: {T["bg"]} !important;
    color: {T["text"]} !important;
    font-family: 'Inter', 'Segoe UI', sans-serif !important;
}}

[data-testid="stHeader"] {{
    background-color: {T["surface"]} !important;
    border-bottom: 1px solid {T["border"]} !important;
}}
[data-testid="stHeader"] * {{
    color: {T["text_muted"]} !important;
}}
[data-testid="stHeader"] button svg {{
    fill: {T["text_muted"]} !important;
}}
[data-testid="stHeader"] button {{
    color: {T["text_muted"]} !important;
    background: transparent !important;
    border: 1px solid {T["border"]} !important;
    border-radius: 6px !important;
}}

footer, #MainMenu {{
    visibility: hidden !important;
    display: none !important;
}}

[data-testid="stMain"],
[data-testid="block-container"],
.main .block-container {{
    background-color: {T["bg"]} !important;
    padding-top: 2rem !important;
}}

[data-testid="stSidebar"],
[data-testid="stSidebarContent"] {{
    background-color: {T["sidebar_bg"]} !important;
    border-right: 1px solid {T["border"]} !important;
}}
[data-testid="stSidebar"] * {{
    color: {T["text_dim"]} !important;
}}
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label {{
    color: {T["text_muted"]} !important;
    font-size: 14px !important;
    font-weight: 500 !important;
    padding: 8px 10px 8px 4px !important;
    border-radius: 8px !important;
    margin-bottom: 2px !important;
    display: flex !important;
    align-items: center !important;
}}
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label > div:first-child {{
    margin-right: 8px !important;
    flex-shrink: 0 !important;
}}
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label:hover {{
    color: {T["accent"]} !important;
    background: {T["sidebar_hover"]} !important;
}}

h1, h2, h3, h4, h5, h6 {{ color: {T["text"]} !important; }}
p {{ color: {T["text_muted"]} !important; }}

[data-testid="stTextInput"] input {{
    background-color: {T["surface"]} !important;
    color: {T["text"]} !important;
    border-color: {T["border"]} !important;
    border-radius: 8px !important;
}}
[data-testid="stTextInput"] input::placeholder {{ color: {T["text_dim"]} !important; }}

div[data-baseweb="select"] > div {{
    background-color: {T["surface"]} !important;
    border-color: {T["border"]} !important;
    border-radius: 8px !important;
    color: {T["text"]} !important;
}}
div[data-baseweb="select"] span {{ color: {T["text"]} !important; }}
div[data-baseweb="popover"] ul {{
    background-color: {T["surface"]} !important;
    border: 1px solid {T["border"]} !important;
}}
div[data-baseweb="popover"] li {{
    color: {T["text"]} !important;
    background-color: {T["surface"]} !important;
}}
div[data-baseweb="popover"] li:hover {{
    background-color: {T["surface2"]} !important;
}}

[data-testid="stExpander"] {{
    background-color: {T["qcard_bg"]} !important;
    border: 1px solid {T["qcard_bd"]} !important;
    border-radius: 10px !important;
    margin-bottom: 8px !important;
}}
[data-testid="stExpander"] summary,
[data-testid="stExpander"] summary p,
[data-testid="stExpander"] summary span {{
    color: {T["text"]} !important;
    font-weight: 500 !important;
}}
[data-testid="stExpander"] svg {{ fill: {T["text_dim"]} !important; }}
[data-testid="stExpander"] > div {{
    background-color: {T["qcard_bg"]} !important;
}}

[data-testid="stAlert"] {{
    background-color: {T["surface2"]} !important;
    color: {T["text_muted"]} !important;
    border: 1px solid {T["border"]} !important;
    border-radius: 8px !important;
}}
[data-testid="stAlert"] p {{ color: {T["text_muted"]} !important; }}

.stButton > button {{
    background-color: {T["surface2"]} !important;
    color: {T["accent"]} !important;
    border: 1px solid {T["border2"]} !important;
    border-radius: 8px !important;
    font-weight: 500 !important;
    font-size: 13px !important;
}}
.stButton > button:hover {{
    background-color: {T["surface"]} !important;
    border-color: {T["accent"]} !important;
}}

.metric-card {{
    background: linear-gradient(135deg, {T["card_grad1"]} 0%, {T["card_grad2"]} 100%);
    border: 1px solid {T["border2"]};
    border-radius: 12px;
    padding: 20px 24px;
    position: relative;
    overflow: hidden;
}}
.metric-card::before {{
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: {T["card_top"]};
}}
.metric-label {{
    color: {T["text_dim"]} !important;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 1.2px;
    text-transform: uppercase;
    margin-bottom: 8px;
}}
.metric-value {{
    color: {T["text"]} !important;
    font-size: 32px;
    font-weight: 700;
    line-height: 1;
    margin-bottom: 4px;
}}
.metric-sub {{
    color: {T["text_dimmer"]} !important;
    font-size: 12px;
    margin-top: 6px;
}}

.section-header {{
    color: {T["section"]} !important;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin: 32px 0 16px 0;
    padding-bottom: 8px;
    border-bottom: 1px solid {T["border"]};
}}

.badge-high {{
    background:{T["badge_high_bg"]}; color:{T["badge_high_fg"]} !important;
    border:1px solid {T["badge_high_bd"]};
    padding:3px 10px; border-radius:20px; font-size:12px; font-weight:600; display:inline-block;
}}
.badge-medium {{
    background:{T["badge_med_bg"]}; color:{T["badge_med_fg"]} !important;
    border:1px solid {T["badge_med_bd"]};
    padding:3px 10px; border-radius:20px; font-size:12px; font-weight:600; display:inline-block;
}}
.badge-low {{
    background:{T["badge_low_bg"]}; color:{T["badge_low_fg"]} !important;
    border:1px solid {T["badge_low_bd"]};
    padding:3px 10px; border-radius:20px; font-size:12px; font-weight:600; display:inline-block;
}}
.badge-unverified {{
    background:{T["badge_unv_bg"]}; color:{T["badge_unv_fg"]} !important;
    border:1px solid {T["badge_unv_bd"]};
    padding:3px 10px; border-radius:20px; font-size:12px; font-weight:600; display:inline-block;
}}

hr {{ border-color: {T["border"]} !important; }}
</style>
""", unsafe_allow_html=True)


# Table helper
def html_table(df: pd.DataFrame):
    """Renders a styled HTML table that respects the current theme."""
    th_style = (
        f"padding:10px 14px; text-align:left; font-size:11px; font-weight:600; "
        f"letter-spacing:0.8px; text-transform:uppercase; color:{T['text_dim']}; "
        f"background:{T['table_header']}; border-bottom:2px solid {T['border']};"
    )
    headers = "".join(f"<th style='{th_style}'>{c}</th>" for c in df.columns)

    rows_html = ""
    for i, (_, row) in enumerate(df.iterrows()):
        row_bg = T["table_row_alt"] if i % 2 == 0 else T["surface"]
        td_style = (
            f"padding:10px 14px; font-size:13px; color:{T['text_muted']}; "
            f"background:{row_bg}; border-bottom:1px solid {T['border']};"
        )
        cells = "".join(f"<td style='{td_style}'>{v}</td>" for v in row)
        rows_html += f"<tr>{cells}</tr>"

    st.markdown(f"""
    <div style="background:{T['surface']}; border:1px solid {T['border']};
                border-radius:10px; overflow:hidden; margin-top:8px;">
        <table style="width:100%; border-collapse:collapse;">
            <thead><tr>{headers}</tr></thead>
            <tbody>{rows_html}</tbody>
        </table>
    </div>
    """, unsafe_allow_html=True)


#Sidebar
with st.sidebar:
    st.markdown(f"""
    <div style='padding:12px 4px 24px 4px;'>
        <div style='font-size:20px; font-weight:700; color:{T["accent"]}; letter-spacing:-0.3px;'>
            ⬡ RAG System
        </div>
        <div style='font-size:11px; color:{T["text_dim"]}; letter-spacing:1.5px;
                    text-transform:uppercase; margin-top:6px; font-weight:500;'>
            Evaluation Dashboard
        </div>
        <div style='margin-top:16px; height:1px;
                    background:linear-gradient(90deg, {T["border"]}, transparent);'></div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        ["Overview", "Hallucination Tracker", "Query Explorer", "Document Manager"],
        label_visibility="collapsed",
    )

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

    mode_label = "☀️  Light mode" if st.session_state.dark_mode else "🌙  Dark mode"
    if st.button(mode_label, use_container_width=True):
        st.session_state.dark_mode = not st.session_state.dark_mode
        st.rerun()

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    if st.button("↺  Refresh data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()



#Chart theme
CHART_THEME = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor=T["chart_bg"],
    font=dict(color=T["text_muted"], family="Inter, sans-serif", size=12),
    xaxis=dict(
        gridcolor=T["chart_grid"],
        linecolor=T["chart_line"],
        tickcolor=T["chart_line"],
        tickfont=dict(color=T["text_muted"]),
        title_font=dict(color=T["text_muted"]),
    ),
    yaxis=dict(
        gridcolor=T["chart_grid"],
        linecolor=T["chart_line"],
        tickcolor=T["chart_line"],
        tickfont=dict(color=T["text_muted"]),
        title_font=dict(color=T["text_muted"]),
    ),
    margin=dict(l=40, r=20, t=30, b=40),
)

CONF_COLORS = {
    "high":       T["green"],
    "medium":     T["amber"],
    "low":        T["red"],
    "unverified": T["gray"],
}

#Data loader
@st.cache_data(ttl=30)
def load_data():
    stats      = get_summary_stats()
    queries    = get_all_queries()
    ingestions = get_all_ingestions()
    df = pd.DataFrame(queries) if queries else pd.DataFrame()
    if not df.empty:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["sources"]   = df["sources"].apply(
            lambda x: json.loads(x) if isinstance(x, str) else x
        )
    return stats, df, ingestions

#Page header
def page_header(title: str, subtitle: str):
    st.markdown(f"""
    <div style='margin-bottom:28px;'>
        <h1 style='color:{T["text"]}; font-size:26px; font-weight:700; margin:0;'>{title}</h1>
        <p style='color:{T["text_dimmer"]}; font-size:13px; margin:6px 0 0 0;'>{subtitle}</p>
    </div>
    """, unsafe_allow_html=True)

# PAGE 1 - OVERVIEW
if page == "Overview":
    stats, df, _ = load_data()
    page_header("System Overview", "Live metrics across all queries and evaluations")

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
        fc = T["green"] if faith >= 0.8 else T["amber"] if faith >= 0.5 else T["red"]
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Avg Faithfulness</div>
            <div class="metric-value" style="color:{fc}">{faith:.2f}</div>
            <div class="metric-sub">RAGAS score (0–1)</div>
        </div>""", unsafe_allow_html=True)

    with c3:
        hal = stats['hallucination_rate']
        hc  = T["green"] if hal < 10 else T["amber"] if hal < 30 else T["red"]
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Hallucination Rate</div>
            <div class="metric-value" style="color:{hc}">{hal}%</div>
            <div class="metric-sub">{stats['hallucination_count']} low-confidence answers</div>
        </div>""", unsafe_allow_html=True)

    with c4:
        lat = stats['avg_latency_ms']
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Avg Latency</div>
            <div class="metric-value">{lat:,.0f}<span style="font-size:16px;color:{T['text_dimmer']}"> ms</span></div>
            <div class="metric-sub">end-to-end per query</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)

    if df.empty:
        st.info("No queries logged yet — ask questions via POST /ask to populate the dashboard.")
        st.stop()

    left, right = st.columns([3, 2])
    with left:
        st.markdown('<div class="section-header">Faithfulness over time</div>', unsafe_allow_html=True)
        valid = df[df["faithfulness_score"] >= 0]
        fig = go.Figure()
        if not valid.empty:
            fig.add_trace(go.Scatter(
                x=valid["timestamp"], y=valid["faithfulness_score"],
                mode="lines+markers",
                line=dict(color=T["blue"], width=2),
                marker=dict(color=T["blue"], size=6, line=dict(color=T["surface"], width=1.5)),
                fill="tozeroy", fillcolor=T["chart_fill"],
                hovertemplate="<b>%{y:.2f}</b><br>%{x|%H:%M %d %b}<extra></extra>",
            ))
        fig.add_hline(y=0.8, line_dash="dash", line_color=T["green"], opacity=0.6,
                      annotation_text="good threshold", annotation_font_color=T["green"])
        fig.update_layout(**CHART_THEME, height=260, showlegend=False)
        fig.update_yaxes(range=[0, 1.05])
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.markdown('<div class="section-header">Confidence distribution</div>', unsafe_allow_html=True)
        counts = stats["confidence_counts"]
        labels, values = list(counts.keys()), list(counts.values())
        colors = [CONF_COLORS.get(l, T["gray"]) for l in labels]
        fig2 = go.Figure(go.Pie(
            labels=labels, values=values,
            marker=dict(colors=colors, line=dict(color=T["pie_outline"], width=2)),
            hole=0.6, textinfo="label+percent",
            textfont=dict(color=T["text_muted"], size=12),
            hovertemplate="<b>%{label}</b><br>%{value} queries<extra></extra>",
        ))
        fig2.update_layout(**CHART_THEME, height=260, showlegend=False,
                           annotations=[dict(text=f"<b>{stats['total_queries']}</b><br>total",
                                             x=0.5, y=0.5, showarrow=False,
                                             font=dict(color=T["text"], size=18))])
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown('<div class="section-header">Latency distribution (ms)</div>', unsafe_allow_html=True)
    fig3 = go.Figure(go.Histogram(
        x=df["latency_ms"], nbinsx=20,
        marker=dict(color=T["purple"], opacity=0.8, line=dict(color=T["bg"], width=1)),
        hovertemplate="<b>%{x:.0f} ms</b><br>%{y} queries<extra></extra>",
    ))
    fig3.update_layout(**CHART_THEME, height=200, bargap=0.1)
    fig3.update_xaxes(title_text="Latency (ms)",
                  title_font=dict(color=T["text_muted"]),
                  tickfont=dict(color=T["text_muted"]))
    fig3.update_yaxes(title_text="Queries",
                  title_font=dict(color=T["text_muted"]),
                  tickfont=dict(color=T["text_muted"]))
    st.plotly_chart(fig3, use_container_width=True)

    st.markdown('<div class="section-header">Recent queries</div>', unsafe_allow_html=True)
    recent = df.head(10)[["timestamp","question","faithfulness_score","confidence_level","latency_ms"]].copy()
    recent["timestamp"]        = recent["timestamp"].dt.strftime("%d %b %H:%M")
    recent["question"]         = recent["question"].str[:65] + "…"
    recent["faithfulness_score"] = recent["faithfulness_score"].apply(
        lambda x: f"{x:.2f}" if x >= 0 else "—"
    )
    recent["latency_ms"]       = recent["latency_ms"].apply(lambda x: f"{x:,.0f} ms")
    recent.columns             = ["Time","Question","Faithfulness","Confidence","Latency"]
    html_table(recent)


# PAGE 2 — HALLUCINATION TRACKER
elif page == "Hallucination Tracker":
    _, df, _ = load_data()
    page_header("Hallucination Tracker", "Faithfulness scores, confidence trends, and flagged answers")

    if df.empty:
        st.info("No queries logged yet.")
        st.stop()

    valid = df[df["faithfulness_score"] >= 0].copy()

    st.markdown('<div class="section-header">Faithfulness score timeline</div>', unsafe_allow_html=True)
    fig = go.Figure()
    fig.add_hrect(y0=0,   y1=0.5, fillcolor=T["red"],   opacity=0.04, line_width=0,
                  annotation_text="danger zone", annotation_font_color=T["red"],   annotation_position="top left")
    fig.add_hrect(y0=0.5, y1=0.8, fillcolor=T["amber"], opacity=0.04, line_width=0,
                  annotation_text="caution",     annotation_font_color=T["amber"], annotation_position="top left")
    fig.add_hrect(y0=0.8, y1=1.0, fillcolor=T["green"], opacity=0.04, line_width=0,
                  annotation_text="good",        annotation_font_color=T["green"], annotation_position="top left")
    if not valid.empty:
        point_colors = [CONF_COLORS.get(c, T["gray"]) for c in valid["confidence_level"]]
        fig.add_trace(go.Scatter(
            x=valid["timestamp"], y=valid["faithfulness_score"],
            mode="lines+markers",
            line=dict(color=T["blue"], width=1.5, dash="dot"),
            marker=dict(color=point_colors, size=10, line=dict(color=T["surface"], width=1.5)),
            hovertemplate="<b>Score: %{y:.2f}</b><br>%{x|%H:%M %d %b}<extra></extra>",
        ))
    fig.update_layout(**CHART_THEME, height=300, showlegend=False)
    fig.update_yaxes(range=[0, 1.05])
    st.plotly_chart(fig, use_container_width=True)

    left, right = st.columns(2)
    with left:
        st.markdown('<div class="section-header">Confidence breakdown</div>', unsafe_allow_html=True)
        conf_counts = df["confidence_level"].value_counts()
        fig2 = go.Figure()
        for level, color in CONF_COLORS.items():
            fig2.add_trace(go.Bar(
                name=level.capitalize(), x=[level.capitalize()],
                y=[conf_counts.get(level, 0)], marker_color=color,
                marker_line=dict(color=T["bg"], width=1),
                hovertemplate=f"<b>{level}</b><br>%{{y}} queries<extra></extra>",
            ))
        fig2.update_layout(**CHART_THEME, height=260, barmode="group", showlegend=True,
                           legend=dict(font=dict(color=T["text_muted"])))
        st.plotly_chart(fig2, use_container_width=True)

    with right:
        st.markdown('<div class="section-header">NLI verdict distribution</div>', unsafe_allow_html=True)
        nli_counts = df["nli_verdict"].value_counts()
        nli_colors_map = {"clean": T["green"], "uncertain": T["amber"], "contradicted": T["red"]}
        fig3 = go.Figure(go.Pie(
            labels=nli_counts.index.tolist(), values=nli_counts.values.tolist(),
            marker=dict(colors=[nli_colors_map.get(l, T["gray"]) for l in nli_counts.index],
                        line=dict(color=T["pie_outline"], width=2)),
            hole=0.55, textinfo="label+percent",
            textfont=dict(color=T["text_muted"], size=12),
            hovertemplate="<b>%{label}</b><br>%{value} queries<extra></extra>",
        ))
        fig3.update_layout(**CHART_THEME, height=260, showlegend=False)
        st.plotly_chart(fig3, use_container_width=True)

    st.markdown('<div class="section-header">Lowest faithfulness answers</div>', unsafe_allow_html=True)
    if valid.empty:
        st.info("No scored answers yet.")
    else:
        worst = valid.nsmallest(5, "faithfulness_score")[
            ["timestamp","question","answer","faithfulness_score","confidence_level"]
        ].copy()
        for _, row in worst.iterrows():
            badge_class = f"badge-{row['confidence_level']}"
            sc = T["green"] if row["faithfulness_score"] >= 0.8 else T["amber"] if row["faithfulness_score"] >= 0.5 else T["red"]
            st.markdown(f"""
            <div style="background:{T['qcard_bg']}; border:1px solid {T['qcard_bd']};
                        border-radius:10px; padding:16px; margin-bottom:10px;">
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
                    <span style="color:{T['text_dim']}; font-size:12px;">
                        {row['timestamp'].strftime('%d %b %Y %H:%M')}
                    </span>
                    <div style="display:flex; gap:8px; align-items:center;">
                        <span style="color:{sc}; font-weight:700; font-size:14px;">
                            {row['faithfulness_score']:.2f}
                        </span>
                        <span class="{badge_class}">{row['confidence_level']}</span>
                    </div>
                </div>
                <div style="color:{T['text']}; font-size:13px; font-weight:500; margin-bottom:6px;">
                    Q: {row['question'][:100]}
                </div>
                <div style="color:{T['text_muted']}; font-size:12px; line-height:1.5;">
                    {row['answer'][:200]}…
                </div>
            </div>
            """, unsafe_allow_html=True)

# PAGE 3 — QUERY EXPLORER
elif page == "Query Explorer":
    _, df, _ = load_data()
    page_header("Query Explorer", "Browse, search, and inspect every question and answer")

    if df.empty:
        st.info("No queries logged yet.")
        st.stop()

    f1, f2, f3 = st.columns([3, 1, 1])
    with f1:
        search = st.text_input("Search", placeholder="Filter questions…", label_visibility="collapsed")
    with f2:
        conf_filter = st.selectbox("Confidence", ["All","high","medium","low","unverified"], label_visibility="collapsed")
    with f3:
        sort_by = st.selectbox("Sort by", ["Newest first","Lowest faithfulness","Highest latency"], label_visibility="collapsed")

    filtered = df.copy()
    if search:
        filtered = filtered[filtered["question"].str.contains(search, case=False, na=False)]
    if conf_filter != "All":
        filtered = filtered[filtered["confidence_level"] == conf_filter]
    if sort_by == "Newest first":
        filtered = filtered.sort_values("timestamp", ascending=False)
    elif sort_by == "Lowest faithfulness":
        filtered = filtered[filtered["faithfulness_score"] >= 0].sort_values("faithfulness_score")
    elif sort_by == "Highest latency":
        filtered = filtered.sort_values("latency_ms", ascending=False)

    st.markdown(f"""
    <div style="color:{T['text_dim']}; font-size:12px; margin:8px 0 16px;">
        Showing {len(filtered)} of {len(df)} queries
    </div>
    """, unsafe_allow_html=True)

    for _, row in filtered.iterrows():
        badge_class = f"badge-{row['confidence_level']}"
        score       = row["faithfulness_score"]
        score_str   = f"{score:.2f}" if score >= 0 else "—"
        score_color = T["green"] if score >= 0.8 else T["amber"] if score >= 0.5 else T["red"] if score >= 0 else T["gray"]
        sources     = row.get("sources", [])
        sources_html = " ".join([
            f'<span style="background:{T["pill_bg"]};color:{T["pill_fg"]};'
            f'padding:2px 8px;border-radius:4px;font-size:11px;margin-right:4px;font-weight:500;">{s}</span>'
            for s in (sources if isinstance(sources, list) else [])
        ])
        with st.expander(f"  {row['question'][:90]}", expanded=False):
            st.markdown(f"""
            <div style="display:flex;gap:12px;align-items:center;margin-bottom:12px;flex-wrap:wrap;">
                <span style="color:{T['text_dim']};font-size:12px;">
                    {pd.to_datetime(row['timestamp']).strftime('%d %b %Y %H:%M')}
                </span>
                <span class="{badge_class}">{row['confidence_level']}</span>
                <span style="color:{score_color};font-weight:600;font-size:13px;">
                    Faithfulness: {score_str}
                </span>
                <span style="color:{T['text_dim']};font-size:12px;">{row['latency_ms']:.0f} ms</span>
            </div>
            <div style="color:{T['text']};font-size:13px;line-height:1.7;margin-bottom:12px;">
                {row['answer']}
            </div>
            <div style="margin-top:8px;">{sources_html}</div>
            """, unsafe_allow_html=True)

# PAGE 4 — DOCUMENT MANAGER
elif page == "Document Manager":
    _, _, ingestions = load_data()
    page_header("Document Manager", "Ingested documents, chunk counts, and ingestion history")

    st.markdown('<div class="section-header">Documents on disk</div>', unsafe_allow_html=True)
    docs_path = Path("data/documents")
    files     = list(docs_path.glob("*.pdf")) + list(docs_path.glob("*.txt"))

    if not files:
        st.info("No documents found in data/documents/")
    else:
        cols = st.columns(3)
        for i, f in enumerate(files):
            size_kb = round(f.stat().st_size / 1024, 1)
            with cols[i % 3]:
                st.markdown(f"""
                <div class="metric-card" style="margin-bottom:12px;">
                    <div class="metric-label">{f.suffix.upper().strip('.')}</div>
                    <div style="color:{T['text']};font-size:14px;font-weight:600;
                                margin-bottom:4px;word-break:break-all;">{f.name}</div>
                    <div class="metric-sub">{size_kb} KB</div>
                </div>
                """, unsafe_allow_html=True)

    st.markdown('<div class="section-header">Ingestion summary</div>', unsafe_allow_html=True)
    if not ingestions:
        st.info("No ingestion events recorded yet.")
    else:
        ing_df = pd.DataFrame(ingestions)
        ing_df["timestamp"] = pd.to_datetime(ing_df["timestamp"])
        summary = (ing_df.groupby("source")
                   .agg(total_chunks=("chunks","max"), runs=("source","count"),
                        last_ingested=("timestamp","max"), last_status=("status","last"))
                   .reset_index())

        for _, row in summary.iterrows():
            sc = T["green"] if row["last_status"] == "success" else T["amber"]
            st.markdown(f"""
            <div style="background:{T['qcard_bg']};border:1px solid {T['qcard_bd']};
                        border-radius:10px;padding:16px;margin-bottom:10px;
                        display:flex;justify-content:space-between;align-items:center;">
                <div>
                    <div style="color:{T['text']};font-size:13px;font-weight:600;">
                        {Path(row['source']).name}
                    </div>
                    <div style="color:{T['text_dim']};font-size:12px;margin-top:4px;">
                        {row['source']}
                    </div>
                </div>
                <div style="display:flex;gap:24px;align-items:center;">
                    <div style="text-align:center;">
                        <div style="color:{T['text']};font-size:18px;font-weight:700;">
                            {int(row['total_chunks'])}
                        </div>
                        <div style="color:{T['text_dim']};font-size:11px;">chunks</div>
                    </div>
                    <div style="text-align:center;">
                        <div style="color:{T['text']};font-size:18px;font-weight:700;">
                            {int(row['runs'])}
                        </div>
                        <div style="color:{T['text_dim']};font-size:11px;">runs</div>
                    </div>
                    <div style="text-align:center;">
                        <div style="color:{sc};font-size:13px;font-weight:600;">
                            {row['last_status']}
                        </div>
                        <div style="color:{T['text_dim']};font-size:11px;">
                            {row['last_ingested'].strftime('%d %b %H:%M')}
                        </div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('<div class="section-header">Full ingestion log</div>', unsafe_allow_html=True)
    if ingestions:
        ing_df["timestamp"] = ing_df["timestamp"].dt.strftime("%d %b %Y %H:%M")
        ing_df["source"]    = ing_df["source"].apply(lambda x: Path(x).name)
        display             = ing_df[["timestamp","source","chunks","status","doc_id"]].copy()
        display.columns     = ["Time","File","Chunks","Status","Doc ID"]
        display["Doc ID"]   = display["Doc ID"].str[:12] + "…"
        html_table(display)