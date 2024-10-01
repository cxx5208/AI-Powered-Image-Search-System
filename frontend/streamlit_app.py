import sys
import os
import streamlit as st
import logging
from PIL import Image, ImageDraw
import logging



# Add the root project directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from app.vector_storage import get_all_labels  # Import the new helper function
# Import suppress warnings
import warnings
from transformers import logging as transformers_logging
warnings.filterwarnings("ignore", message="Some weights of the model checkpoint at")
transformers_logging.set_verbosity_error()  # Suppress Hugging Face specific logs

# Import your custom functions
from app.object_detection import detect_objects
from app.vector_storage import store_image_vector, load_vectors  # Add load_vectors for object retrieval
from app.rag_retrieval import retrieve_similar_images
from app.image_utils import save_image

# Set up logging
logging.basicConfig(filename='logs/app.log', level=logging.INFO)

# Apply custom CSS
def apply_custom_css():
    with open('frontend/custom.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

apply_custom_css()

# Set upload directory
UPLOAD_DIR = "data/images"
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

# Streamlit App with advanced UI
st.title("🚀 Object Detection & Image Search System")

# Main layout using columns for cleaner structure
col1, col2 = st.columns([2, 3])

with col1:
    st.markdown("### 📂 Upload an Image")
    uploaded_file = st.file_uploader("", type=["png", "jpg", "jpeg"])

    if uploaded_file:
        # Save uploaded image
        save_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
        save_image(uploaded_file, save_path)

        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

        try:
            # Detect objects in the image
            bounding_boxes, labels, scores = detect_objects(save_path)

            # Draw bounding boxes on the image
            image = Image.open(save_path)
            draw = ImageDraw.Draw(image)

            detected_objects = []
            for box, label, score in zip(bounding_boxes, labels, scores):
                draw.rectangle(box.tolist(), outline="red", width=2)
                draw.text((box[0], box[1]), f"Label: {label}, Score: {score:.2f}", fill="red")
                detected_objects.append({'box': box.tolist(), 'label': label, 'score': float(score)})

            # Display the image with bounding boxes
            st.image(image, caption="Detected Objects", use_column_width=True)

            # Store detected object data in vector database
            store_image_vector(uploaded_file.name, detected_objects)
            st.success("✅ Objects detected and stored in vector database.")
            logging.info(f"Processed image {uploaded_file.name}")

        except Exception as e:
            st.error(f"Error processing the image: {str(e)}")
            logging.error(f"Failed to process image {uploaded_file.name}: {str(e)}")

with col2:
    st.markdown("### 🔎 Search for Object in Stored Images")
    
    # Get all unique labels from the vector storage
    all_labels = get_all_labels()
    st.write(f"Available labels in the database: {all_labels}")

    object_names = st.multiselect(
        "Enter object names to search for (multiple allowed)", 
        options=all_labels,  # Dynamically set options based on detected labels
        help="You can select multiple objects to search for images containing all selected objects."
    )

    if st.button("Find Images with Objects"):
        if object_names:
            st.write(f"### Images containing the objects: {', '.join(object_names)}:")
            found_images = retrieve_similar_images(object_names)  # Pass list of object names
            if found_images:
                for img in found_images:
                    img_path = os.path.join(UPLOAD_DIR, img)
                    st.image(img_path, caption=f"{img} containing {' & '.join(object_names)}", use_column_width=True)
            else:
                st.write("No images found for the selected objects.")
        else:
            st.error("Please select at least one object name.")


# Footer section
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("## Made with 💻 using Streamlit", unsafe_allow_html=True)
