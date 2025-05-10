#%%
import torch
class CharTokenizer:
    def __init__(self):
        self.vocab = ["<pad>", "<unk>"]

    @property
    def vocab_size(self): return len(self.vocab)

    def build_vocab(self, text): return self.vocab.extend(set(text) - set(self.vocab))

    def encode(self, text): return [self.vocab.index(char) if char in self.vocab else "<unk>" for char in text]

    def __call__(self, text, truncation=True, padding='max_length', max_length=None, return_tensors=None):
        ids = self.encode(text)

        # トランケーション
        if truncation and len(ids) > max_length:
            ids = ids[:max_length]

        # パディング
        if padding == 'max_length':
            ids += [self.vocab.index("<pad>")] * (max_length - len(ids))

        output = {"input_ids": ids}

        # Tensor化
        if return_tensors == 'pt':
            output["input_ids"] = torch.tensor(output["input_ids"]).unsqueeze(0)  # (1, max_length)

        return output

    def decode(self, ids):
        if isinstance(ids, torch.Tensor):
            ids = ids.squeeze().tolist()
        return "".join([self.vocab[i] if 0<=i<self.vocab_size else "<unk>" for i in ids])

