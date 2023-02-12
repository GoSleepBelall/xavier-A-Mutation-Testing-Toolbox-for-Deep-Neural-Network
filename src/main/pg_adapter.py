import psycopg2
import os
from dotenv import load_dotenv, find_dotenv

class PgAdapter:
    __instance__ = None

    def __init__(self):
        if PgAdapter.__instance__ is None:
            PgAdapter.__instance__ = self
            load_dotenv(find_dotenv())
            self.connection = psycopg2.connect(
                database=os.getenv("DB_NAME"),
                user=os.getenv("DB_USER"),
                password=os.getenv("DB_PASS"),
                host=os.getenv("DB_URL"),
                port=os.getenv("DB_PORT"),
            )

    @staticmethod
    def get_instance():
        # We define the static method to fetch instance
        if not PgAdapter.__instance__:
            PgAdapter()
        return PgAdapter.__instance__