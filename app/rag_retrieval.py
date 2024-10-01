from app.vector_storage import query_faiss_index

def retrieve_similar_images(object_names):
    """
    Use the stored vectors to retrieve similar images containing all specified objects.
    """
    return query_faiss_index(object_names)
