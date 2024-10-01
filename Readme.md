# 🚀 AI-Powered Object Detection & Image Search System

## Overview

This project is an **AI-powered object detection and image search system** built using cutting-edge **computer vision** models and a **vector-based retrieval system**. The core idea is to allow users to upload images, detect objects using a state-of-the-art object detection model (Facebook's DETR), and then store object information in a **vector database**. Users can search for images containing one or more objects, and the system will retrieve all matching images from the stored database using **Retrieval-Augmented Generation (RAG)**.

## Features

- **Upload Images**: Users can upload images in **PNG**, **JPEG**, or **JPG** format.
- **Object Detection**: The system detects multiple objects in uploaded images using **Facebook's DETR (DEtection TRansformers)** model.
- **Bounding Boxes and Scores**: Each detected object is highlighted with bounding boxes, along with labels and confidence scores.
- **Dynamic Object Search**: Users can search for images containing one or more selected objects. The system only returns images that contain **all** selected objects.
- **Advanced UI**: Modern, responsive user interface with **custom CSS**, animations, and tooltips.
- **Retrieval-Augmented Generation (RAG)**: Used for intelligent and efficient image search and retrieval from the vector database.

## Technologies Used

- **Streamlit**: Interactive, web-based interface for seamless user interaction.
- **Transformers (Hugging Face)**: For integrating state-of-the-art AI models like DETR and RAG.
- **DEtection TRansformers (DETR)**: Facebook's DETR model is used for object detection, which leverages a transformer-based architecture.
- **Retrieval-Augmented Generation (RAG)**: Combines the power of retrieval-based models and generative models for efficient and intelligent search.
- **PIL (Python Imaging Library)**: Image processing library used for handling images (drawing bounding boxes, saving images, etc.).
- **FAISS**: Facebook's AI Similarity Search is used to perform fast vector searches over large datasets.
- **JSON**: To store vector data in an easily accessible format.
- **Faiss Index**: To efficiently search over stored vectors based on detected objects.
- **Python**: The core language for backend development and AI model integration.
- **CSS**: Custom styles to enhance the UI with transitions, animations, and better interactivity.

## Algorithms & Models Used

### 1. **Object Detection - Facebook's DETR**
The **DEtection TRansformers (DETR)** model is used to detect objects in uploaded images. It uses a transformer-based architecture that combines:
- **Convolutional Neural Networks (CNNs)** for feature extraction.
- **Transformers** for object detection and bounding box prediction.
- **Bounding Box Regression** and **Label Classification**: After extracting features, DETR classifies and generates bounding boxes for the objects it detects.

The output includes:
- **Bounding boxes** (coordinates of detected objects).
- **Labels** (e.g., "Car", "Person", "Dog", etc.).
- **Confidence scores** for each detected object.

### 2. **Retrieval-Augmented Generation (RAG)**
RAG combines:
- **Dense retrieval** using FAISS (Facebook AI Similarity Search).
- **Generative model** that leverages transformers to retrieve similar images based on object vectors.

In this system:
- **Vectors**: After detecting objects, we convert the bounding boxes and labels into vectors using FAISS.
- **Retrieval**: The system retrieves images based on these vectors, efficiently finding images that contain the same or similar objects as the query.
- **Augmentation**: The system augments the retrieval process by combining vector-based search with label-based search using object names.

### 3. **FAISS (Facebook AI Similarity Search)**
- **Vector Database**: All detected objects are stored as vectors in a JSON-based vector database. 
- **Vector Matching**: FAISS is used to efficiently search for similar images based on object vectors. FAISS can handle large datasets with high-dimensional data, making it ideal for this project.

### How RAG Is Used
- **Object-based Search**: Once an image is uploaded and processed, object vectors are generated.
- **RAG Pipeline**: When a user searches for images, the system combines vector-based similarity search with label-based search using RAG to retrieve images that match all the specified objects.
- **Efficient Retrieval**: The combination of FAISS and RAG ensures that the search process is both fast and accurate, even when querying multiple objects.

## Architecture

```
+--------------------+        +-----------------+        +--------------------+
|   Frontend (UI)     | <----> |   Backend (API) | <----> |   Object Detection  |
+--------------------+        +-----------------+        +--------------------+
      |                                    |                             |
      v                                    v                             v
+--------------------+        +-----------------+        +--------------------+
|  Upload Image       |       |  RAG Retrieval   |       |  DETR (Object Det.) |
+--------------------+        +-----------------+        +--------------------+
      |                                    |                             |
      v                                    v                             v
+--------------------+        +-----------------+        +--------------------+
|  Store Vectors      |        |  FAISS Indexing |       |  Object Recognition|
+--------------------+        +-----------------+        +--------------------+
```

## Installation & Setup

To run this project locally, follow the steps below.

### Prerequisites

- Python 3.x
- Pip (Python package installer)

### Install Dependencies

Clone this repository and navigate to the project directory:

```bash
git clone https://github.com/your-username/AI_EDGE_RAG.git
cd AI_EDGE_RAG
```

Install the required Python packages:

```bash
pip install -r requirements.txt
```

### Running the Application

Once all dependencies are installed, run the Streamlit app:

```bash
streamlit run frontend/streamlit_app.py
```

The app should open in your browser at `http://localhost:8501`.

### Project Structure

```bash
AI_EDGE_RAG/
├── app/
│   ├── object_detection.py         # DETR object detection logic
│   ├── vector_storage.py           # Vector storage and retrieval using FAISS
│   ├── rag_retrieval.py            # Retrieval-Augmented Generation logic
│   ├── image_utils.py              # Image saving utilities
├── data/
│   ├── images/                     # Uploaded images storage
│   └── vectors/                    # Stored image vectors (JSON format)
├── frontend/
│   ├── custom.css                  # Custom CSS for UI styling
│   ├── streamlit_app.py            # Streamlit application code
├── logs/
│   └── app.log                     # Logging output
├── tests/                          # Unit tests for individual modules
├── requirements.txt                # Python package dependencies
└── README.md                       # Project README
```

### Sample Workflow

1. **Upload Image**: Upload any image (PNG, JPG, JPEG).
2. **Object Detection**: The system will detect objects, display the image with bounding boxes, and store the object data in the vector database.
3. **Search by Object**: You can search for images containing one or more objects. The system will use RAG to retrieve images based on object vectors and labels.
4. **Results**: Only images containing **all** selected objects will be displayed.

## Future Enhancements

- **GPU Support**: Leverage GPU for faster inference times on large image datasets.
- **Real-Time Image Processing**: Add support for live image feeds from a camera.
- **Deployment**: Deploy the app to a cloud service like **AWS** or **Heroku** for global accessibility.

## Contributions

Feel free to submit issues or pull requests. Contributions are welcome!
