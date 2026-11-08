import networkx as nx

def build_graph(focus_movie_id: str, movie_service, max_related_movies: int = 20):
    focus_movie = movie_service.get_movie_by_id(focus_movie_id)

    if focus_movie is None:
        return nx.Graph()

    related_movies = movie_service.get_related_movies(focus_movie_id, limit=max_related_movies)

    graph = nx.Graph()

    def add_movie(movie):
        movie_node = f"movie:{movie.id}"

        graph.add_node(movie_node, label=str(movie.title), kind="Movie", overview=str(movie.overview), year=str(movie.year), movie_id=movie.id,)

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

    add_movie(focus_movie)

    for movie in related_movies:
        add_movie(movie)

    return graph