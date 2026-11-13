
import streamlit as st
import networkx as nx
import plotly.graph_objects as go
import json
from pathlib import Path
import tempfile

from services.movie_service import MovieService
from services.query_cache import (get_cached_title_search, get_cached_movie_graph)

def render_graph_component(fig):
    node_trace_index = len(fig.data) - 1

    node_ids = list(fig.data[node_trace_index].customdata)

    edge_pairs = [
        (trace.meta["source"], trace.meta["target"])
        for trace in fig.data[:node_trace_index]
    ]

    adjacency = {node: [] for node in node_ids}

    for u, v in edge_pairs:
        adjacency[u].append(v)
        adjacency[v].append(u)

    node_ids_json = json.dumps(node_ids)
    adjacency_json = json.dumps(adjacency)
    edge_pairs_json = json.dumps(edge_pairs)

    html = fig.to_html(full_html=True, include_plotlyjs="cdn")

    script = f"""
    <script>
    const gd = document.querySelector('.plotly-graph-div');

    const nodeIds = {node_ids_json};
    const adjacency = {adjacency_json};
    const edgePairs = {edge_pairs_json};

    const nodeTraceIndex = {node_trace_index};
    const edgeIndices = Array.from(
        {{length: nodeTraceIndex}},
        (_, i) => i
    );

    function resetGraph() {{
        Plotly.restyle(gd, {{
            'marker.opacity': [nodeIds.map(() => 1)]
        }}, [nodeTraceIndex]);

        Plotly.restyle(gd, {{
            'opacity': edgePairs.map(() => 0.55)
        }}, edgeIndices);
    }}

    gd.on('plotly_hover', function(event) {{
        const point = event.points.find(
            p => p.curveNumber === nodeTraceIndex
        );

        if (!point) return;

        const hoveredNode = point.customdata;

        const connected = new Set([
            hoveredNode,
            ...(adjacency[hoveredNode] || [])
        ]);

        const nodeOpacity = nodeIds.map(node =>
            connected.has(node) ? 1 : 0.08
        );

        const edgeOpacity = edgePairs.map(([u, v]) =>
            u === hoveredNode || v === hoveredNode ? 0.8 : 0.03
        );

        Plotly.restyle(gd, {{
            'marker.opacity': [nodeOpacity]
        }}, [nodeTraceIndex]);

        Plotly.restyle(gd, {{
            'opacity': edgeOpacity
        }}, edgeIndices);
    }});

    gd.on('plotly_unhover', resetGraph);
    </script>
    """
    html = html.replace(
        "</head>",
        """
        <style>
            html, body {
                margin: 0;
                padding: 0;
                overflow: hidden !important;
            }

            .plotly-graph-div {
                width: 100% !important;
            }
        </style>
        </head>
        """
    )

    html = html.replace("</body>", script + "</body>")

    st.iframe(html, height=620)

@st.cache_data(show_spinner=False)
def get_cached_graph_figure(_graph, focus_movie_id: str, dataset_fingerprint: str):
    return make_graph_figure(_graph, focus_movie_id=focus_movie_id)

def make_graph_figure(graph, focus_movie_id, max_nodes=100):
    focus = f"movie:{focus_movie_id}"

    if focus not in graph:
        return go.Figure()

    visible = {focus}
    neighbors = set(graph.neighbors(focus))

    priorities = ["Movie", "Director", "Genre", "Actor"]

    for kind in priorities:
        candidates = sorted(node for node in neighbors if graph.nodes[node].get("kind") == kind)
        remaining = max_nodes - len(visible)

        if remaining <= 0:
            break

        visible.update(candidates[:remaining])

    entity_nodes = {node for node in visible if graph.nodes[node].get("kind") != "Movie"}
    related_movies = set()

    for entity in entity_nodes:
        related_movies.update(node for node in graph.neighbors(entity) if graph.nodes[node].get("kind") == "Movie")

    remaining = max_nodes - len(visible)

    if remaining > 0:
        visible.update(sorted(related_movies - visible)[:remaining])

    subgraph = graph.subgraph(visible).copy()

    if not subgraph:
        return go.Figure()

    pos = nx.spring_layout(subgraph, seed=17, k=0.9, iterations=20)

    edge_traces = []

    for u, v, attrs in subgraph.edges(data=True):
        x0, y0 = pos[u]
        x1, y1 = pos[v]

        edge_trace = go.Scatter(
            x=[x0, x1],
            y=[y0, y1],
            mode="lines",
            line=dict(width=1, color="#a8a8a8"),
            opacity=0.55,
            hoverinfo="skip",
            showlegend=False,
            meta={"source": u, "target": v}
        )

        edge_traces.append(edge_trace)

    color_map = {
        "Movie": "#4776b4",
        "Actor": "#35a16b",
        "Director": "#d98b32",
        "Genre": "#9467bd"
    }

    node_x = []
    node_y = []
    node_text = []
    node_colors = []
    node_sizes = []
    node_ids = []

    for node, attrs in subgraph.nodes(data=True):
        x, y = pos[node]

        kind = attrs.get("kind", "Entity")
        label = attrs.get("label", node)

        node_ids.append(node)
        node_x.append(x)
        node_y.append(y)

        node_text.append(f"{label}<br>Type: {kind}")

        node_colors.append("#e24a4a" if node == focus else color_map.get(kind, "#777777"))
        node_sizes.append(25 if node == focus else 17 if kind == "Movie" else 12)

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        text=[attrs.get("label", node) for node, attrs in subgraph.nodes(data=True)],
        textposition="top center",
        hovertext=node_text,
        hoverinfo="text",
        customdata=node_ids,
        marker=dict(
            size=node_sizes,
            color=node_colors,
            opacity=1,
            line=dict(width=1, color="white")
        ),
        showlegend=False,
        meta="nodes"
    )

    fig = go.Figure(data=edge_traces + [node_trace])

    fig.update_layout(
        height=600,
        margin=dict(l=10, r=10, t=20, b=10),
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        hovermode="closest"
    )

    return fig

def render_graph_legend():
    color_map = {
        "Movie": "#4776b4",
        "Actor": "#35a16b",
        "Director": "#d98b32",
        "Genre": "#9467bd",
        "Focus": "#e24a4a"
    }

    legend_items = "".join(
        f"""
        <div style="display:flex; align-items:center; gap:5px;">
            <span style="width:9px; height:9px; border-radius:50%; background:{color}; flex-shrink:0;"></span>
            <span style="font-size:12px;">{label}</span>
        </div>
        """
        for label, color in color_map.items()
    )

    st.html(f"""
        <div style="
            padding:9px 12px;
            border:1px solid rgba(128,128,128,0.35);
            border-radius:8px;
            margin-bottom:10px;
        ">
            <div style="font-size:13px; font-weight:600; margin-bottom:8px;">
                Node Legend
            </div>
            <div style="display:flex; flex-wrap:wrap; align-items:center; gap:16px;">
                {legend_items}
            </div>
        </div>
    """)

@st.fragment
def render_graph_page(movie_service: MovieService, embeddings, dataset_fingerprint: str):
    st.subheader("Explore movie connections")

    title_query = st.text_input("Find a movie by title", placeholder="Start typing a movie title...", key="graph_movie_search")

    matching_movies = get_cached_title_search(movie_service, title_query, 20, dataset_fingerprint)

    if not matching_movies:
        if title_query.strip():
            st.info("No matching movie titles found.")
        else:
            st.info("Enter a movie title to explore its connections.")
        return

    movies_by_id = {movie.id: movie for movie in matching_movies}

    selected_id = st.selectbox(
        "Matching movies",
        options=list(movies_by_id.keys()),
        format_func=lambda movie_id: (
            f"{movies_by_id[movie_id].title} "
            f"({movies_by_id[movie_id].year})"
        ),
        key=f"graph_movie_{title_query.casefold()}"
    )

    if selected_id:
        render_graph_legend()
        graph = get_cached_movie_graph(movie_service, embeddings, selected_id, dataset_fingerprint)

        fig = get_cached_graph_figure(graph, selected_id, dataset_fingerprint)

        render_graph_component(fig)