# csv_to_faiss.py
import pandas as pd
import numpy as np
import faiss
import json
from openai import OpenAI

CSV_FILE = "data.csv"  # your CSV file

df = pd.read_csv(CSV_FILE)
texts = [" ".join(str(x) for x in row.values) for _, row in df.iterrows()]

client = OpenAI(api_key="YOUR_OPENAI_KEY")

embeddings = []
for text in texts:
    emb = client.embeddings.create(model="text-embedding-3-small", input=text)
    embeddings.append(emb.data[0].embedding)

embeddings = np.array(embeddings).astype("float32")

index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

faiss.write_index(index, "my_vectors.faiss")
with open("my_docs.json", "w") as f:
    json.dump(texts, f)

print("✅ Vector database created: my_vectors.faiss + my_docs.json")
