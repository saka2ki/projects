#%%
from torch.utils.data import Dataset, DataLoader
from src.notebook.charTokenizer import CharTokenizer
import src.notebook.path as path
import json

class JPNDataset(Dataset):
    def __init__(self, path, tokenizer, max_length):
        super().__init__()
        self.path = path
        self.tokenizer = tokenizer
        self.max_length = max_length

        # 各行のファイル位置オフセットだけ記録（高速アクセス用）
        self.offsets = []
        with open(self.path, "r", encoding="utf-8") as f:
            offset = f.tell()
            line = f.readline()
            while line:
                self.offsets.append(offset)
                offset = f.tell()
                line = f.readline()


    def __len__(self): return len(self.offsets)
    def __getitem__(self, idx):
         with open(self.path, "r", encoding="utf-8") as f:
            f.seek(self.offsets[idx])
            line = f.readline()
            data = json.loads(line)
            text = data.get("text", "")
            tokens = self.tokenizer(text, max_length=self.max_length, return_tensors="pt")
            return tokens["input_ids"].squeeze(0)