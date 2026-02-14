from PIL import Image
import chromadb
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
from chromadb.utils.data_loaders import ImageLoader
import numpy as np
from tqdm import tqdm
import os
import pillow_heif

db_path=r"C:\Users\leona\OneDrive\Desktop\progetto\chromadb_ViT-H-14" #add your db path here

# Initialize Chroma DB client, embedding function, and data loader
client = chromadb.PersistentClient(path=db_path)
embedding_function = OpenCLIPEmbeddingFunction(model_name = "ViT-H-14-378-quickgelu", checkpoint= "dfn5b", device= "cuda")
data_loader = ImageLoader()

collection = client.get_or_create_collection(
    name='multimodal_collection3',
    embedding_function=embedding_function,
    data_loader=data_loader
)

def add_images_to_collection(folder_path):
    image_files = [os.path.join(folder_path, image_name) for image_name in os.listdir(folder_path)
                   if os.path.isfile(os.path.join(folder_path, image_name)) and image_name.lower().endswith(('.png', '.jpg', '.jpeg', '.heic', '.webp', 'heif'))]

    for image_path in tqdm(image_files,desc="Creating Image Embeddings and Adding to DB"):
        try:
            image = np.array(Image.open(image_path))
            collection.add(
                ids=[os.path.basename(image_path)],
                images=[image]
            )
        except Exception as e:
            print(f"Error processing {image_path}: {e}")

image_folder_path=r"C:\Users\leona\OneDrive\Desktop\progetto\images_for_indexing\docs" #add your folder path here

add_images_to_collection(image_folder_path)

