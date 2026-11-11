import re
from patterns import SPLIT_PATTERN
from pathlib import Path

def get_stats(chunks,counts=None):
    counts = {} if counts is None else counts
    for chunk in chunks:
        for pair in zip(chunk, chunk[1:]):
            counts[pair] = counts.get(pair, 0) + 1
    return counts


def merge(ids, pair, rep):
    new_ids=[]
    i=0
    while i < len(ids):
        if i< len(ids)-1 and ids[i]==pair[0] and ids[i+1]==pair[1]:
            new_ids.append(rep)
            i+=2
        else:
            new_ids.append(ids[i])
            i+=1
    return new_ids


class Tokenizer:
    def __init__(self):
        self.merges={}
        self.vocabs={idx:bytes([idx]) for idx in range(256)}
        self.special_tokens = {}
        self.inverse_special_tokens = {}
        self.special_pattern = None
        
    def train(self, text, vocab_size, verbose=True):
            assert vocab_size > 255,"vocab_size must be at least 256 (base 1-byte tokens)."
            num_merges=vocab_size-256 

            chunks=re.findall(SPLIT_PATTERN, text)
            chunk_ids=[list(chunk.encode("utf-8")) for chunk in chunks]
            for i in range(num_merges):
                stats=get_stats(chunk_ids)
                if not stats:
                    break
                pair=max(stats, key=stats.get)
                if stats[pair] < 2:
                    break
                idx= 256+i
                chunk_ids=[merge(chunk,pair,idx) for chunk in chunk_ids]  
                self.merges[pair]=idx
                self.vocabs[idx]=self.vocabs[pair[0]]+ self.vocabs[pair[1]]
                if verbose and ((i + 1) % 50 == 0 or i == num_merges - 1):
                     print(f"Merge {i+1}/{num_merges}: {pair} -> {idx} ({self.vocabs[idx]!r}) had {stats[pair]} occurrences")

    def encode(self, text, allowed_special=None):

        if not self.special_pattern or allowed_special is None:
            return self._encode_ordinary(text)
        tokens=[]
        parts=re.split(self.special_pattern, text)

        for part in parts:
            if part in self.special_tokens:
                tokens.append(self.special_tokens[part])
            else:
                tokens.extend(self._encode_ordinary(part))
        return tokens

    def decode(self, flat_ids):
        raw_byte=b"".join(self.vocabs.get(id_, b'?') for id_ in flat_ids)
        return raw_byte.decode("utf-8", errors="replace")
        
    def register_special_tokens(self,special_tokens):
        self.special_tokens=special_tokens
        self.inverse_special_tokens={v:k for k,v in special_tokens.items()}

        if special_tokens:
            for token_str, token_id in special_tokens.items():
                self.vocabs[token_id]=token_str.encode("utf-8")
            # just preparing where to split and the brackets are to return the mathes too, since by defualt split removes them.
            pattern = "(" + "|".join(re.escape(k) for k in special_tokens) + ")"
            self.special_pattern=re.compile(pattern)
        else:
            self.special_pattern=None

    def _encode_ordinary(self, text):
        chunks = re.findall(SPLIT_PATTERN, text)
        chunk_ids = [list(chunk.encode("utf-8")) for chunk in chunks]

        while True:
            stats = get_stats(chunk_ids)
            if not stats:
                break
            pair = min(stats.keys(), key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            chunk_ids = [merge(chunk, pair, self.merges[pair]) for chunk in chunk_ids]

        return [id_ for chunk in chunk_ids for id_ in chunk]

    def save(self, file_prefix):
        parent=Path(__file__).resolve().parent
        model_path=parent/"data"/f"{file_prefix}.model"
        vocab_path=parent/"data"/f"{file_prefix}.vocab"
        model_path.parent.mkdir(parents=True, exist_ok=True)

        with open(model_path, 'w', encoding="utf-8") as f_m:
            f_m.write("bpe_model v1\n")
            f_m.write(f"{len(self.special_tokens)}\n")
            for token, idx in self.special_tokens.items():
                f_m.write(f"{token}  {idx}\n")
            for p0,p1 in self.merges:
                f_m.write(f"{p0}  {p1}\n")

        with open(vocab_path, 'w', encoding="utf-8") as f_v:
            f_v.write("bpe_vocab v1\n")
            for idx, token_bytes in sorted(self.vocabs.items()):
                rendered=token_bytes.decode("utf-8", errors="replace")
                f_v.write(f"[{idx}]  {token_bytes!r} -> {rendered}\n")
            

    def load(self,model_path):
            model_path=Path(model_path)
            assert model_path.exists(), f"model file not found at {model_path}"

    
            self.vocabs={idx: bytes([idx]) for idx in range(256)}
            with open(model_path, 'r', encoding="utf-8") as f_m:
                head=f_m.readline().strip()
                assert head =="bpe_model v1", "Invalid model file header!"
                special_num=int(f_m.readline().strip())
               
                special_tokens={}
                for _ in range(special_num):
                    token,idx=f_m.readline().strip().split()
                    special_tokens[token]=int(idx)
                if special_tokens:
                    self.register_special_tokens(special_tokens)
                idx=256
                for line in f_m:
                    line= line.strip()
                    if not line:
                        continue
                    p0,p1=map(int, line.split())
                    self.merges[(p0,p1)]=idx
                    self.vocabs[idx]=self.vocabs[p0] + self.vocabs[p1]
                    idx+=1

   

                    


                    



        


        







    
        
