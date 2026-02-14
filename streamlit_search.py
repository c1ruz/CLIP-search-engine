import streamlit as st
import chromadb
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
from chromadb.utils.data_loaders import ImageLoader
import os

db_path=r"C:\Users\leona\OneDrive\Desktop\progetto\chromadb_modellobase" #add your db path here

# Initialize Chroma DB client, embedding function, and data loader
client = chromadb.PersistentClient(path=db_path)
embedding_function = OpenCLIPEmbeddingFunction()
data_loader = ImageLoader()

collection = client.get_or_create_collection(
    name='multimodal_collection3',
    embedding_function=embedding_function,
    data_loader=data_loader
)

# Display the banner image
banner_image_path = r"C:\Users\leona\OneDrive\Desktop\progetto\images_for_indexing\docs\image-149.jpg"  # Update with the path to your banner image
st.image(banner_image_path)


st.title("Test for Omer Image Search Engine")
st.write("Number of elements in the collection:", collection.count())




# Search bar
query = st.text_input("Enter your search query: ")
parent_path = r"C:\Users\leona\OneDrive\Desktop\progetto\images_for_indexing\docs" #add your image folder path here
if st.button("Search"):
    results = collection.query(query_texts=[query], n_results=5,include=["distances"])
    print(results)
    for image_id, distance in zip(results['ids'][0], results['distances'][0]):
        image_path = os.path.join(parent_path, image_id)
        st.image(image_path, caption=os.path.basename(image_path))
        st.write(f"Distance: {distance}")