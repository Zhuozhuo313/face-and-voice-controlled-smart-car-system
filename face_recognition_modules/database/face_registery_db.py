from . import database
import uuid
from face_recognition_modules.configs import global_config

class FaceRegisteryDatabse(database.Database):
    def __init__(self, unknow=False) -> None:
        super().__init__(global_config.registery_db)
        self.table_name = "person" if not unknow else "unknow"
        # create table if not exists
        self.execute(
            f"CREATE TABLE IF NOT EXISTS {self.table_name} (id TEXT PRIMARY KEY, person_id TEXT, name TEXT, embedding BLOB)"
        )
        self.commit()
    
    def add_person(self, person_name, embedding, person_id=None):
        face_id = str(uuid.uuid4())
        if person_id is None:
            person_id = face_id
        self.execute(f"INSERT INTO {self.table_name} VALUES (?, ?, ?, ?)", (face_id, person_id, person_name, embedding))
        self.commit()
        return face_id
    
    def get_person_name(self, face_id):
        cursor = self.execute(f"SELECT name FROM {self.table_name} WHERE id=?", (face_id,))
        for row in cursor:
            return row[0]
        return None
    
    def get_person_id(self, face_id):
        cursor = self.execute(f"SELECT person_id FROM {self.table_name} WHERE id=?", (face_id,))
        for row in cursor:
            return row[0]
        return None
    
    def clear(self):
        self.execute(f"DELETE FROM {self.table_name}")
        self.commit()
    
    def get_persons(self):
        cursor = self.execute(f"SELECT * FROM {self.table_name}")
        for row in cursor:
            yield row
