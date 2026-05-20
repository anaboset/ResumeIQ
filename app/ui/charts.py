import plotly.graph_objects as go
import pandas as pd
from plotly.colors import hex_to_rgb

def score_gauge(score: float, title: str = "", height: int = 180) -> go.Figure:
    """Create a gauge chart for a score."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score * 100,
        title={"text": title, "font": {"size": 13, "color": "#94a3b8"}},
        number={"suffix": "%", "font": {"size": 22, "color": "#f8fafc"}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#475569", "tickfont": {"color": "#475569"}},
            "bar": {"color": "#fbbf24", "thickness": 0.35},
            "bgcolor": "#1e293b",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 40], "color": "#1e293b"},
                {"range": [40, 55], "color": "#1e3a2e"},
                {"range": [55, 75], "color": "#1e3a2e"},
                {"range": [75, 100], "color": "#1e3a2e"},
            ],
            "threshold": {
                "line": {"color": "#22c55e", "width": 2},
                "thickness": 0.75,
                "value": 75,
            },
        },
    ))
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=30, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": "#f8fafc"},
    )
    return fig


def signal_bar_chart(signals: dict) -> go.Figure:
    """Horizontal bar chart for signal breakdown."""
    labels = [v["label"] for v in signals.values()]
    scores = [v["score"] * 100 for v in signals.values()]
    weights = [v["weight"] * 100 for v in signals.values()]
    weighted = [v["weighted"] * 100 for v in signals.values()]
    colors = ["#fbbf24", "#34d399", "#818cf8"]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=labels,
        x=scores,
        orientation="h",
        marker_color=colors,
        name="Signal Score",
        text=[f"{s:.1f}%" for s in scores],
        textposition="outside",
        textfont={"color": "#f8fafc", "size": 12},
    ))
    fig.update_layout(
        height=180,
        margin=dict(l=10, r=40, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(range=[0, 115], showgrid=False, zeroline=False,
                   tickfont={"color": "#64748b"}),
        yaxis=dict(showgrid=False, tickfont={"color": "#cbd5e1"}),
        showlegend=False,
        font={"color": "#f8fafc"},
    )
    return fig


def comparison_radar(df: pd.DataFrame, top_n: int = 5) -> go.Figure:
    """Radar chart comparing top N candidates across signals."""
    top = df.head(top_n)
    categories = ["Semantic", "Skill Match", "Experience", "Overall"]

    fig = go.Figure()
    colors = ["#fbbf24", "#34d399", "#818cf8", "#f472b6", "#60a5fa"]
    

    for i, (_, row) in enumerate(top.iterrows()):
        values = [
            row["semantic_score"] * 100,
            row["skill_match_score"] * 100,
            row["experience_score"] * 100,
            row["final_score"] * 100,
        ]
        color = colors[i % len(colors)]
        rgb = hex_to_rgb(color)
        values.append(values[0])  # close polygon
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories + [categories[0]],
            fill="toself",
            name=row["name"][:20],
            line_color=colors[i % len(colors)],
            fillcolor=f"rgba({rgb[0]}, {rgb[1]}, {rgb[2]}, 0.08)",
            opacity=0.85,
        ))

    fig.update_layout(
        polar=dict(
            bgcolor="#1e293b",
            radialaxis=dict(visible=True, range=[0, 100],
                            tickfont={"size": 9, "color": "#64748b"},
                            gridcolor="#334155"),
            angularaxis=dict(tickfont={"size": 11, "color": "#cbd5e1"},
                             gridcolor="#334155"),
        ),
        showlegend=True,
        legend=dict(font={"color": "#cbd5e1"}, bgcolor="rgba(0,0,0,0)"),
        paper_bgcolor="rgba(0,0,0,0)",
        height=380,
        margin=dict(l=40, r=40, t=30, b=30),
        font={"color": "#f8fafc"},
    )
    return fig


def tier_donut(summary: dict) -> go.Figure:
    """Donut chart for tier breakdown."""
    tiers = ["Strong Match", "Good Match", "Partial Match", "Weak Match"]
    counts = [
        summary.get("strong_matches", 0),
        summary.get("good_matches", 0),
        summary.get("partial_matches", 0),
        summary.get("weak_matches", 0),
    ]
    colors = ["#22c55e", "#84cc16", "#f59e0b", "#ef4444"]

    fig = go.Figure(go.Pie(
        labels=tiers,
        values=counts,
        hole=0.6,
        marker_colors=colors,
        textinfo="label+value",
        textfont={"size": 11, "color": "#f8fafc"},
        hovertemplate="%{label}: %{value} candidate(s)<extra></extra>",
    ))
    fig.update_layout(
        height=260,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        font={"color": "#f8fafc"},
    )
    return fig
