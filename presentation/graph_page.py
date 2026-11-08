
import streamlit as st
import networkx as nx
import plotly.graph_objects as go

from services.movie_service import MovieService
from services.query_cache import (get_cached_title_search, get_cached_movie_graph)

@st.cache_data(show_spinner=False)
def get_cached_graph_figure(_graph, focus_movie_id: str, dataset_fingerprint: str):
    return make_graph_figure(_graph, focus_movie_id=focus_movie_id)

def make_graph_figure(graph, focus_movie_id, max_nodes=65):
    focus = f"movie:{focus_movie_id}"

    if focus not in graph:
        return go.Figure()

    visible = {focus}

    visible.update(graph.neighbors(focus))

    for neighbor in list(visible):
        if graph.nodes[neighbor].get("kind") != "Movie":
            visible.update(node for node in graph.neighbors(neighbor) if graph.nodes[node].get("kind") == "Movie")

    visible.discard(focus)
    visible = {focus, *list(visible)[:max_nodes - 1]}

    subgraph = graph.subgraph(visible).copy()

    if subgraph.number_of_nodes() == 0:
        return go.Figure()

    pos = nx.spring_layout(subgraph, seed=17, k=0.9, iterations=50)

    edge_x = []
    edge_y = []

    for u, v in subgraph.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]

        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(x=edge_x, y=edge_y, mode="lines", line=dict(width=1, color="#a8a8a8"), hoverinfo="none", showlegend=False)

    color_map = {"Movie": "#4776b4", "Actor": "#35a16b", "Director": "#d98b32", "Genre": "#9467bd"}

    node_x = []
    node_y = []
    node_text = []
    node_colors = []
    node_sizes = []

    for node, attrs in subgraph.nodes(data=True):
        x, y = pos[node]

        kind = attrs.get("kind", "Entity")
        label = attrs.get("label", node)

        node_x.append(x)
        node_y.append(y)

        node_text.append(f"{label}<br>Type: {kind}")

        node_colors.append("#e24a4a" if node == focus else color_map.get(kind, "#777777"))

        node_sizes.append(25 if node == focus else 17 if kind == "Movie" else 12)

    node_trace = go.Scatter(x=node_x, y=node_y, mode="markers+text",
        text=[attrs.get("label", node) for node, attrs in subgraph.nodes(data=True)],
        textposition="top center",
        hovertext=node_text,
        hoverinfo="text",
        marker=dict(size=node_sizes, color=node_colors, line=dict(width=1, color="white")),
        showlegend=False
    )

    fig = go.Figure(data=[edge_trace, node_trace])

    fig.update_layout(
        height=600,
        margin=dict(l=10, r=10, t=20, b=10),
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False)
    )

    return fig

@st.fragment
def render_graph_page(movie_service: MovieService, dataset_fingerprint: str):
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
        graph = get_cached_movie_graph(movie_service, selected_id, dataset_fingerprint)

        fig = get_cached_graph_figure(graph, selected_id, dataset_fingerprint)

        st.plotly_chart(fig, width="stretch")