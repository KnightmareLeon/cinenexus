
class MovieService:
    def __init__(self, movie_repository):
        self.movie_repository = movie_repository

    def get_movie_count(self):
        return self.movie_repository.count_movies()

    def get_movies_page(self, page=1, page_size=50):
        return self.movie_repository.get_movies_page(page=page, page_size=page_size)

    def iter_movies(self, batch_size=500):
        return self.movie_repository.iter_movies(batch_size=batch_size)

    def get_movie_by_id(self, movie_id):
        return self.movie_repository.get_movie_by_id(movie_id)

    def search_movies_by_title(self, query, limit=20):
        return self.movie_repository.search_by_title(query=query, limit=limit)

    def get_movies_by_ids(self, movie_ids):
        return self.movie_repository.get_movies_by_ids(movie_ids)

    def get_related_movies(self, movie_id, limit=5):
        return self.movie_repository.get_related_movies(movie_id=movie_id, limit=limit)

    def get_movie_embedding_index(self, movie_id):
        return self.movie_repository.get_movie_embedding_index(movie_id)

    def get_movie_ids_by_embedding_indices(self, embedding_indices):
        return self.movie_repository.get_movie_ids_by_embedding_indices(embedding_indices)

    def get_embedding_metadata(self):
        return self.movie_repository.get_embedding_metadata()

    def get_search_texts(self):
        return self.movie_repository.get_search_texts()

    def iter_search_texts(self):
        return self.movie_repository.iter_search_texts()
