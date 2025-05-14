#%%
from torch.utils.data import Dataset, DataLoader
from charTokenizer import CharTokenizer
import path as path
import json

class JPNDataset(Dataset):
    def __init__(self, path, tokenizer, max_seq_len):
        super().__init__()
        self.path = path
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

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
            tokens = self.tokenizer(text, max_seq_len=self.max_seq_len, return_tensors="pt")
            return tokens["input_ids"].squeeze(0)