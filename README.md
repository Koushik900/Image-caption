# Image Caption Generator 

An end-to-end deep learning project that generates descriptive captions for images using an Encoder-Decoder architecture with attention mechanism. This model combines a Convolutional Neural Network (CNN) to understand the image features and a Recurrent Neural Network (RNN) with attention to generate fluent captions.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.12%2B-orange?logo=tensorflow)
![Keras](https://img.shields.io/badge/Keras-%23D00000.svg?logo=keras&logoColor=white)

## Features

*   **Encoder-Decoder Architecture:** Uses a pre-trained CNN (InceptionV3) as an encoder and an LSTM network as a decoder.
*   **Attention Mechanism:** Implements Bahdanau Attention to allow the decoder to focus on relevant parts of the image while generating each word.
*   **Customizable Training:** Easy-to-follow Jupyter notebooks for data preprocessing, model training, and inference.
*   **Web Demo (Optional):** Includes a simple Streamlit app for a interactive user interface to upload images and get captions.
*   **BLEU Score Evaluation:** Scripts to evaluate the quality of the generated captions against human references.

## Project Structure
```plaintext
Image-caption/
│── data/                # Dataset files (images + captions)
│── notebooks/           # Jupyter/Colab notebooks
│── models/              # Saved trained models
│── utils/               # Helper functions (data preprocessing, evaluation, etc.)
│── requirements.txt     # Dependencies
│── README.md            # Project documentation
│── main.py              # Main training/testing script
```

## Quick Start

### Prerequisites

*   Python 3.8+
*   pip

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Koushik900/Image-caption.git
    cd Image-caption
    ```

2.  **(Recommended) Create a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### Downloading the Dataset

This project is designed to work with the **Flickr8k** dataset. Due to licensing, you need to download it manually.

1.  Visit the [Flickr8k Dataset website](https://forms.illinois.edu/sec/1713398) and request access.
2.  Download the dataset and extract it.
3.  Place the `Flickr8k_Data/` folder (containing images) and the `Flickr8k_text/` folder (containing captions) in the project's root directory.

### Running the Code

The easiest way to run the project is through the provided Jupyter notebooks, which guide you through the entire process.

1.  **Preprocess the Data:**
    Open and run `notebooks/1_data_preprocessing.ipynb`. This will tokenize the captions and prepare the image features.

2.  **Train the Model:**
    Open and run `notebooks/2_model_training.ipynb`. This will define the model architecture and start the training process. You can modify hyperparameters like batch size, number of epochs, and embedding dimension here.

3.  **Generate Captions!**
    Open and run `notebooks/3_inference.ipynb`. Upload an image and see the magic happen.

### Trying the Web Demo

1.  Ensure you have a trained model saved in the `models/` directory.
2.  Navigate to the `app/` directory and run the Streamlit app:
    ```bash
    streamlit run streamlit_app.py
    ```
3.  Open your browser to the local URL shown in the terminal (usually `http://localhost:8501`).

## Model Architecture

1.  **Encoder:** A pre-trained **InceptionV3** model (with final classification layer removed) is used to extract feature vectors from images. These features are flattened and passed to the decoder.
2.  **Decoder:** An **LSTM** network generates the caption one word at a time.
3.  **Attention:** The **Bahdanau Attention** layer helps the decoder decide which parts of the image to focus on for each word generation step, significantly improving caption quality.

## 📊 Results & Evaluation

The model is evaluated using the **BLEU (Bilingual Evaluation Understudy) Score**, a standard metric for evaluating the quality of machine-generated text against human-written references.

After training for 20 epochs, the model typically achieves:
*   **BLEU-1:** ~0.60 - 0.65
*   **BLEU-2:** ~0.40 - 0.45
*   **BLEU-3:** ~0.25 - 0.30
*   **BLEU-4:** ~0.15 - 0.20

*Note: Results may vary based on hyperparameters and training time.*

**Example Prediction:**
| | |
| :--- | :--- |
| **Input Image** | <img src="https://github.com/user-attachments/assets/.../example.jpg" width="300"/> |
| **Generated Caption** | *"A little girl is smiling and playing in the water."* |
| **Reference Captions** | *"A young girl in a blue dress is running through the water."* <br> *"A child is playing in a fountain on a sunny day."* |

## Future Improvements

*   Experiment with larger datasets like **Flickr30k** or **MS-COCO**.
*   Try different encoder architectures (**VGG19**, **ResNet50**, **EfficientNet**).
*   Implement a **Transformer-based** architecture for image captioning.
*   Add **Beam Search** during inference for better caption generation.
*   Deploy the model as a full web application using **Flask/Django** and **Heroku/AWS**.

## Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](https://github.com/Koushik900/Image-caption/issues) or open a new one.

1.  Fork the Project
2.  Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3.  Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4.  Push to the Branch (`git push origin feature/AmazingFeature`)
5.  Open a Pull Request



##  Acknowledgments

*   The Flickr8k dataset creators and providers.
*   TensorFlow and Keras teams for excellent documentation and tutorials.
*   Inspired by various research papers and online tutorials on image captioning and attention mechanisms.

## 📧 Contact

Koushik - tkkoushikinin@gmail.com

Project Link: [https://github.com/Koushik900/Image-caption](https://github.com/Koushik900/Image-caption)
