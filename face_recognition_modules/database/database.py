import sqlite3
from typing import Any
class Database:
    def __init__(self, db_path):
        self.db = sqlite3.connect(db_path)
    
    def execute(self, query: str, params: Any=())->sqlite3.Cursor:
        cursor = self.db.execute(query, params)
        return cursor

    def commit(self):
        self.db.commit()
    
    def clear(self):
        pass
