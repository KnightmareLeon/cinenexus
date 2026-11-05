import networkx as nx

from domain.movie import Movie

def build_graph(movies: list[Movie]):
    graph = nx.Graph()

    for movie in movies:
        movie_node = f"movie:{movie.id}"

        graph.add_node(
            movie_node,
            label=str(movie.title),
            kind="Movie",
            overview=str(movie.overview),
            year=str(movie.year),
            movie_id=movie.id,
        )

        for genre in movie.genres:
            node = f"genre:{genre}"

            graph.add_node(node, label=genre, kind="Genre")

            graph.add_edge(movie_node, node, relation="has genre")

        if movie.director:
            node = f"director:{movie.director}"

            graph.add_node(node, label=movie.director, kind="Director")

            graph.add_edge(movie_node, node, relation="directed by")

        for actor in movie.actors:
            node = f"actor:{actor}"

            graph.add_node(node, label=actor, kind="Actor")

            graph.add_edge(movie_node, node, relation="features")

    return graph