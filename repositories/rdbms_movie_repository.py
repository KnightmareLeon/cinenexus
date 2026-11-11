from abc import ABC, abstractmethod

class RDBMSMovieRepository(ABC):

    @abstractmethod
    def get_movie_by_id(self, movie_id: str):
        pass

    @abstractmethod
    def get_movies_by_ids(self, movie_ids: list[str]):
        pass

    @abstractmethod
    def get_movies_page(self, page: int = 1, page_size: int = 50):
        pass

    @abstractmethod
    def iter_movies(self, batch_size: int = 500):
        """Yield catalog movies in display order without loading them all at once."""
        pass

    @abstractmethod
    def count_movies(self) -> int:
        pass

    @abstractmethod
    def search_by_title(self, query: str, limit: int = 20):
        pass

    @abstractmethod
    def get_related_movies(self, movie_id: str, limit: int = 20):
        pass

    @abstractmethod
    def get_movie_embedding_index(self, movie_id: str) -> int | None:
        pass

    @abstractmethod
    def get_movie_ids_by_embedding_indices(self, embedding_indices: list[int]) -> list[str]:
        pass

    @abstractmethod
    def get_embedding_metadata(self) -> dict[str, str]:
        pass

    @abstractmethod
    def get_search_texts(self) -> list[str]:
        pass

    @abstractmethod
    def iter_search_texts(self):
        pass
