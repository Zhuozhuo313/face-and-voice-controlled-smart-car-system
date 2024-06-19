import faiss
import numpy as np

class FeatureSearch:
    def __init__(self,  feature_dim: int = 512) -> None:
        self.feature_dim = feature_dim
        index = faiss.IndexFlatIP(feature_dim)
        self.feature_lib = faiss.IndexIDMap(index)
        self.index2id = {}
        self.lib_index = 0
    
    def search(self, feature: np.ndarray):
        if len(self.index2id) == 0:
            return "", 0
        distances, indices = self.feature_lib.search(feature, 1)
        if len(distances) == 0:
            return "", 0
        person_id = self.index2id[indices[0][0]]
        score = distances[0][0]
        return person_id, score
    
    def add(self, feature: np.ndarray, id: str):
        self.feature_lib.add_with_ids(feature, np.array([self.lib_index]))
        self.index2id[self.lib_index] = id
        self.lib_index += 1
    
    def clear(self):
        self.feature_lib.reset()
        self.index2id.clear()
        self.lib_index = 0
    
if __name__=="__main__":
    face_searcher = FeatureSearch()
    face_searcher.add(np.random.rand(1,512), "1")
    face_searcher.add(np.random.rand(1,512), "2")
    face_searcher.add(np.random.rand(1,512), "3")

    distances, indices = face_searcher.search(np.random.rand(1,512), 3)
    print(distances)
    print(indices)
    