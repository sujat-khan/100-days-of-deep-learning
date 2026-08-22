# Day 20: Keras Functional API & Multi-Output Models

This directory covers the **Keras Functional API**, an essential paradigm in Deep Learning for designing complex neural network architectures beyond the linear layer-by-layer stack of `keras.Sequential`.

---

## 📌 Resource Links & Notebook Overviews

### 1. [Notebook 1: Keras Functional API Fundamentals](https://colab.research.google.com/drive/1gtXbh7X9IB1AjCymcYwLpEYyMxSmfOsS#scrollTo=D99PWX9yegtL)
* **Colab Link:** [Open in Google Colab](https://colab.research.google.com/drive/1gtXbh7X9IB1AjCymcYwLpEYyMxSmfOsS#scrollTo=D99PWX9yegtL)
* **Local Notebook:** [`functional_api_demo.ipynb`](./functional_api_demo.ipynb)
* **Description & Key Concepts:**
  - **Introduction to Functional Syntax:** How layers are called directly as functions with tensors as inputs and outputs (`output = Layer()(input)`).
  - **Explicit Input Layer:** Defining model inputs via `Input(shape=(...))`.
  - **Multi-Output Architecture:** Constructing a model with a shared feature extractor and two distinct output heads:
    - *Output 1 (Regression):* Linear activation for continuous target prediction.
    - *Output 2 (Binary Classification):* Sigmoid activation for probability prediction.
  - **Model Inspection & Visualization:** Using `model.summary()` and `keras.utils.plot_model(model, show_shapes=True)` to inspect connectivity and layer flow.

---

### 2. [Notebook 2: Age & Gender Prediction (End-to-End Multi-Output Project)](https://colab.research.google.com/drive/1B2azaSX9g55olY473c7WAMf3E5-717Ge?usp=sharing#scrollTo=71CoXULRw73J)
* **Colab Link:** [Open in Google Colab](https://colab.research.google.com/drive/1B2azaSX9g55olY473c7WAMf3E5-717Ge?usp=sharing#scrollTo=71CoXULRw73J)
* **Local Notebook:** [`age_gender_revised.ipynb`](./age_gender_revised.ipynb)
* **Description & Key Concepts:**
  - **Dataset Setup:** Automated download and extraction of the **UTKFace** dataset via Kaggle API (`jangedoo/utkface-new`).
  - **Data Parsing & Preprocessing:**
    - Extracting `age` and `gender` target labels directly from image filenames (`[age]_[gender]_[race]_[date].jpg`).
    - Splitting 23,700+ images into training and testing sets with Pandas.
    - Multi-output data pipeline using `ImageDataGenerator.flow_from_dataframe()` with `class_mode='multi_output'`.
  - **Transfer Learning Backbone:** Leveraging pre-trained **ResNet50** (ImageNet weights) with frozen base layers as the shared feature extractor.
  - **Multi-Task Heads:**
    - **Age Branch:** Dense layers ending in `Dense(1, activation='linear', name='age')` (Mean Squared Error / MAE).
    - **Gender Branch:** Dense layers ending in `Dense(1, activation='sigmoid', name='gender')` (Binary Cross-Entropy / Accuracy).
  - **Joint Multi-Loss Training:** Compiling and optimizing multiple losses simultaneously with individual loss weights and metrics.

---

### 3. [Notebook 3: Age & Gender Multi-Output Model (Alternative / Interactive Colab)](https://colab.research.google.com/drive/1B2azaSX9g55olY473c7WAMf3E5-717Ge?usp=sharing#scrollTo=71CoXULRw73J)
* **Direct Colab Link:** [Open in Google Colab](https://colab.research.google.com/drive/1B2azaSX9g55olY473c7WAMf3E5-717Ge?usp=sharing#scrollTo=71CoXULRw73J)
* **Description & Key Concepts:**
  - Mirror/shared interactive Colab notebook with GPU runtime for running and experimenting with UTKFace multi-task learning.

---

### 4. [Recommended Reading: Keras Functional API Deep Dive](https://machinelearningmastery.com/keras-functional-api-deep-learning/)
* **Article URL:** [Machine Learning Mastery - Keras Functional API for Deep Learning](https://machinelearningmastery.com/keras-functional-api-deep-learning/)
* **Summary & Core Takeaways:**
  - **Sequential vs. Functional:** Why the Sequential API fails for non-linear network topologies (e.g., skip connections, shared layers, multiple inputs/outputs).
  - **Building Blocks:** Input layers, connecting layers, creating `Model(inputs=..., outputs=...)`, and model plotting.
  - **Common Architectural Patterns:**
    1. **Multi-Input Models:** Combining diverse data types (e.g., numerical features + image features).
    2. **Multi-Output Models:** Multi-task learning (predicting multiple attributes from a single sample).
    3. **Shared Layer Models:** Sharing weights across different feature streams (e.g., Siamese networks).
    4. **Residual & Skip Connections:** Implementing ResNet-style identity mapping shortcuts.


