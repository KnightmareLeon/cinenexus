from domain.movie import Movie

def get_movie_by_id(movies: list[Movie], movie_id: str):

    return next((movie for movie in movies if movie.id == str(movie_id)), None)


def get_movie_by_title(movies: list[Movie], title: str):
    return next((movie for movie in movies if movie.title == title), None)

def split_entities(value):
    return tuple(item.strip() for item in str(value).split("|") if item.strip())

def movie_from_row(row):

    director = str(row["director"]).strip()

    return Movie(
        id=str(row["id"]),
        title=str(row["title"]),
        year=str(row["year"]),
        overview=str(row["overview"]),
        genres=split_entities(row["genres"]),
        director=director or None,
        actors=split_entities(row["actors"])
    )