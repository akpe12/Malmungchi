import jsonlines

class KorCorpus:
    def __init__(self, data, readable) -> None:
        self._data = data
        self._readable = readable
        
    @classmethod
    def load(cls, file_path:str):
        data = []
        
        with jsonlines.open(file_path, 'r') as f:
            for line in f:
                data.append(line)
            
        return cls(data, data)
    
    def __getitem__(self, index: int) -> dict:
        index_data = self._readable[index]
        sentence1, sentence3, output = index_data["input"]["sentence1"], index_data["input"]["sentence3"], index_data["output"]
        
        return{
            "sentence1" : sentence1,
            "sentence3" : sentence3,
            "output" : output,
        }
    
    def __len__(self):
        return len(self._data)