import networkx as nx

from services.recommendation_service import get_similar_movies
from services.movie_service import MovieService

def build_graph(focus_movie_id: str, movie_service: MovieService, embeddings, max_related_movies: int = 20):
    focus_movie = movie_service.get_movie_by_id(focus_movie_id)

    if focus_movie is None:
        return nx.Graph()

    related_matches = get_similar_movies(focus_movie_id, movie_service, embeddings, top_k=max_related_movies)

    graph = nx.Graph()
    focus_node = f"movie:{focus_movie.id}"

    def add_movie(movie):
        movie_node = f"movie:{movie.id}"

        graph.add_node(movie_node, label=str(movie.title), kind="Movie", overview=str(movie.overview), year=str(movie.year), movie_id=movie.id)

        for genre in movie.genres or ():
            node = f"genre:{genre}"
            graph.add_node(node, label=genre, kind="Genre")
            graph.add_edge(movie_node, node, relation="has genre")

        if movie.director:
            node = f"director:{movie.director}"
            graph.add_node(node, label=movie.director, kind="Director")
            graph.add_edge(movie_node, node, relation="directed by")

        for actor in movie.actors or ():
            node = f"actor:{actor}"
            graph.add_node(node, label=actor, kind="Actor")
            graph.add_edge(movie_node, node, relation="features")

    add_movie(focus_movie)

    for match in related_matches:
        related_movie = match.movie
        related_node = f"movie:{related_movie.id}"

        add_movie(related_movie)

        graph.add_edge(focus_node, related_node, relation="similar to", similarity=float(match.similarity))

    return graph