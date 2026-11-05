import streamlit as st
import networkx as nx
import plotly.graph_objects as go


def make_graph_figure(graph, focus_movie_id, max_nodes=65):
    focus = f"movie:{focus_movie_id}"

    if focus not in graph:
        return go.Figure()

    visible = {focus}
    visible.update(graph.neighbors(focus))

    for neighbor in list(visible):
        if graph.nodes[neighbor].get("kind") != "Movie":
            visible.update(n for n in graph.neighbors(neighbor) if graph.nodes[n].get("kind") == "Movie")

    visible = set(list(visible)[:max_nodes])
    subgraph = graph.subgraph(visible).copy()

    if not subgraph.nodes:
        return go.Figure()

    pos = nx.spring_layout(subgraph, seed=17, k=0.9)

    edge_x, edge_y = [], []

    for u, v in subgraph.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]

        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        mode="lines",
        line=dict(width=1, color="#a8a8a8"),
        hoverinfo="none",
        showlegend=False
    )

    color_map = {
        "Movie": "#4776b4",
        "Actor": "#35a16b",
        "Director": "#d98b32",
        "Genre": "#9467bd",
    }

    node_x, node_y = [], []
    node_text, node_colors, node_sizes = [], [], []

    for node, attrs in subgraph.nodes(data=True):
        x, y = pos[node]

        node_x.append(x)
        node_y.append(y)

        label = attrs.get("label", node)
        kind = attrs.get("kind", "Entity")

        node_text.append(f"{label}<br>Type: {kind}")

        node_colors.append(
            "#e24a4a" if node == focus
            else color_map.get(kind, "#777777")
        )

        node_sizes.append(25 if node == focus else (17 if kind == "Movie" else 12))

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        text=[graph.nodes[n].get("label", n) for n in subgraph.nodes()],
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
        yaxis=dict(visible=False),
    )

    return fig


def render_graph_page(movies, graph):
    st.subheader("Explore movie connections")

    selected_id = st.selectbox(
        "Select a movie",
        options=[movie.id for movie in movies],
        format_func=lambda movie_id: next(
            movie.title
            for movie in movies
            if movie.id == movie_id
        ),
        key="graph_movie_selector"
    )

    if selected_id:
        fig = make_graph_figure(
            graph,
            focus_movie_id=selected_id
        )

        st.plotly_chart(fig, width="stretch")