# Retriever.py

import os
import pytesseract
from pdf2image import convert_from_path
from sentence_transformers import CrossEncoder
import torch

# -------------------------
# 1️⃣ Tesseract Path Setup (Windows)
# -------------------------
# Update yahan apni tesseract.exe ki path ke sath
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# -------------------------
# 2️⃣ PDF Input
# -------------------------
pdf_path = input("Enter the full path of PDF: ")

if not os.path.exists(pdf_path):
    print("File not found!")
    exit()

# -------------------------
# 3️⃣ Convert PDF pages to images
# -------------------------
pages = convert_from_path(pdf_path)
print(f"Total pages: {len(pages)}")

# -------------------------
# 4️⃣ Extract text chunks from pages
# -------------------------
chunk_size = 2000  # characters per chunk
text_chunks = []

for page in pages:
    text = pytesseract.image_to_string(page, config="--psm 6")
    # Split text into chunks
    for i in range(0, len(text), chunk_size):
        chunk = text[i:i+chunk_size].strip()
        if chunk:
            text_chunks.append(chunk)

print(f"Extracted {len(text_chunks)} text chunks")

# -------------------------
# 5️⃣ Load CrossEncoder Reranker
# -------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print("📦 Loading cross-encoder reranker...")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-12-v2", device=device)

# -------------------------
# 6️⃣ Rank chunks based on relevance
# -------------------------
query = input("Enter your query: ")

# Create input pairs (query, chunk)
pairs = [[query, chunk] for chunk in text_chunks]

# Predict relevance scores
scores = reranker.predict(pairs)

# Sort chunks by score descending
ranked_chunks = [chunk for _, chunk in sorted(zip(scores, text_chunks), key=lambda x: x[0], reverse=True)]

print("\n--- Top k Relevant Chunks ---")
for i, chunk in enumerate(ranked_chunks[:3], 1):
    print(f"\n[{i}] Score: {scores[text_chunks.index(chunk)]:.4f}")
    print(chunk)
