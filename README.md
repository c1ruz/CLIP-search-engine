📸 Multimodal Image Search Engine

This project implements a multimodal image search engine using:

- ChromaDB for vector storage

- OpenCLIP (ViT-H-14) for image-text embeddings

- Streamlit for interactive UI

- CUDA acceleration (optional but recommended)

The system allows you to:

1. Index images into a vector database

2. Perform semantic image search using natural language queries

⚙️ Requirements

Install dependencies:
```bash
pip install chromadb streamlit pillow numpy tqdm pillow-heif
```


If using GPU:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

⚠️ Make sure CUDA is properly configured if using device="cuda"

Important Configuration

Modify these paths:
```python
db_path = r"YOUR_DB_PATH"
image_folder_path = r"YOUR_IMAGE_FOLDER"
```

Then run:
```bash
python add_images.py
```
Run the App
```bash
streamlit run streamlit_search.py
```

🛠 Customization

Change Number of Search Results

In streamlit_search.py:
```python
results = collection.query(query_texts=[query], n_results=5)
```

Modify n_results as needed.

Change Model

You can modify:

```python
OpenCLIPEmbeddingFunction(model_name=..., checkpoint=...)
```

Ensure both scripts use identical embedding configuration.
