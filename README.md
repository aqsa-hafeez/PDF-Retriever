# AI-Powered Cross-Encoder PDF Retriever 🔍📄

A high-precision document retrieval system that combines OCR with Neural Reranking to find the most relevant information within any PDF document.

## 🌟 Key Features
* **Cross-Encoder Scoring**: Uses the `ms-marco-MiniLM-L-12-v2` model to provide deep semantic ranking of document chunks.
* **OCR-Integrated Pipeline**: Utilizes Tesseract OCR and Poppler to process both digital and scanned PDF pages.
* **GPU Acceleration**: Automatically detects and uses NVIDIA CUDA for lightning-fast inference.
* **Precise Retrieval**: Unlike standard keyword search, this system understands the intent behind your query.

## 🛠️ Tech Stack
* **Language**: Python
* **Reranking Model**: Cross-Encoder (Sentence-Transformers)
* **OCR Engine**: Tesseract OCR
* **PDF Processing**: `pdf2image` (Poppler)
* **Hardware**: CUDA supported (optional)

## 📋 How It Works
1. **Conversion**: The PDF is converted into high-quality images page by page.
2. **OCR Extraction**: Tesseract scans each image to extract raw text content.
3. **Chunking**: Text is split into manageable segments (2000 characters each).
4. **Semantic Ranking**: 
   - The user enters a query.
   - The Cross-Encoder pairs the query with every chunk.
   - It assigns a score (0 to 1) based on how well the chunk answers the query.
5. **Output**: The system displays the top-ranked chunks with their relevance scores.

## 💻 Installation & Setup

1. **Install System Dependencies**:
   - Install [Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki).
   - Install [Poppler](https://github.com/oschwartz10612/poppler-windows/releases).

2. **Install Python Libraries**:
   ```bash
   pip install sentence-transformers pytesseract pdf2image torch

```

3. **Configure Paths**:
Update the `tesseract_cmd` path in `Retriever.py` to match your local installation.
4. **Run the Script**:
```bash
python Retriever.py

```



## 📊 Why Use a Cross-Encoder?

Standard retrievers (Bi-Encoders) are fast but sometimes lose accuracy. Cross-Encoders are significantly more powerful because they perform full self-attention over the query and the document chunk simultaneously, making them ideal for high-stakes information retrieval.

---
