
import math
import networkx as nx

from domain.movie import Movie


def build_movie_relationship_index(movies: list[Movie]):
    entity_to_movies = {}

    for movie in movies:
        entities = []

        entities.extend(("genre", genre) for genre in movie.genres)

        if movie.director:
            entities.append(("director", movie.director))

        entities.extend(("actor", actor) for actor in movie.actors)

        for entity_type, name in entities:
            key = (entity_type, name.casefold())

            entity_to_movies.setdefault(key, []).append(movie.id)

    return entity_to_movies


def build_graph(movies: list[Movie], focus_movie_id: str, relationship_index: dict, max_related_movies: int = 20, max_candidates_per_entity: int = 200):
    movie_by_id = {movie.id: movie for movie in movies}

    focus_movie = movie_by_id.get(str(focus_movie_id))

    if focus_movie is None:
        return nx.Graph()

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

    entity_keys = []

    entity_keys.extend(("genre", genre.casefold()) for genre in focus_movie.genres)

    if focus_movie.director:
        entity_keys.append(("director", focus_movie.director.casefold()))

    entity_keys.extend(("actor", actor.casefold()) for actor in focus_movie.actors)

    candidate_scores = {}

    for key in entity_keys:
        related_ids = relationship_index.get(key, [])

        if len(related_ids) > max_candidates_per_entity:
            step = math.ceil(len(related_ids) / max_candidates_per_entity)
            related_ids = related_ids[::step]

        for movie_id in related_ids:
            if movie_id == focus_movie.id:
                continue

            candidate_scores[movie_id] = (candidate_scores.get(movie_id, 0) + 1)

    ranked_ids = sorted(candidate_scores, key=candidate_scores.get, reverse=True)[:max_related_movies]

    for movie_id in ranked_ids:
        movie = movie_by_id.get(movie_id)

        if movie is not None:
            add_movie(movie)

    return graph
