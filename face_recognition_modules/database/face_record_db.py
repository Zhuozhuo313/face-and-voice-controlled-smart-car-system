from . import database
from face_recognition_modules.configs import global_config

class FaceRecordDatabse(database.Database):
    def __init__(self) -> None:
        super().__init__(global_config.record_db)
        # create table if not exists
        self.execute(
            "CREATE TABLE IF NOT EXISTS record (id TEXT PRIMARY KEY, name TEXT,  time TEXT, count INTEGER)"
        )
        self.commit()

    def get_person_record_count(self, person_id):
        cursor = self.execute("SELECT * FROM record WHERE id=?", (person_id,))
        record = cursor.fetchone()
        if record is None:
            return 0
        return record[3]

    def update_record(self, person_id, time):
        cursor = self.execute("SELECT * FROM record WHERE id=?", (person_id,))
        record = cursor.fetchone()
        if record is None:
            self.execute("INSERT INTO record VALUES (?, ?, ?, ?)", (person_id, "", time, 1))
        else:
            count = record[3] + 1
            self.execute("UPDATE record SET count=?, time=? WHERE id=?", (count, time, person_id))
        self.commit()
    
    def clear(self):
        self.execute("DELETE FROM record")
        self.commit()


    