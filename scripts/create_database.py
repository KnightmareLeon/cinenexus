
from pathlib import Path

from repositories.database_repository import DatabaseRepository


ROOT_DIR = Path(__file__).resolve().parent.parent
DATABASE_PATH = ROOT_DIR / "data" / "cinenexus.db"


def main():
    database = DatabaseRepository(DATABASE_PATH)
    database.initialize()

    print("CineNexus database initialized successfully.")
    print(f"Database location: {DATABASE_PATH}")


if __name__ == "__main__":
    main()