# CVI620 Final Project - Group 2

# Video Demo

## Project Overview & Approach
This project implements an autonomous driving system capable of steering a car in the Udacity Self-Driving Car Simulator. We utilized a Convolutional Neural Network (CNN) to predict steering angles based on camera input.

**Our Approach:**
*   **Data Preprocessing:** We applied rigorous image preprocessing including cropping (removing sky/hood), converting to YUV color space, Gaussian blurring, and resizing to match the model's input requirements.
*   **Model Architecture:** We trained a deep learning model (CNN) to map raw pixels to steering commands.
*   **Inference:** A Flask-based Socket.IO server (`TestSimulation.py`) communicates with the simulator in real-time to drive the car autonomously.

## Challenges & Solutions
The most significant hurdle we encountered during training was **Gradient Explosion**, which caused the model weights to become unstable and the loss to diverge.

**How we addressed it:**
To stabilize the training process, we implemented a combination of regularization and optimization techniques:
*   ** Dropout:** Added dropout layers to prevent overfitting and reduce reliance on specific neurons.
*   ** Learning Rate Scheduling:** We reduced the initial learning rate and implemented a `ReduceLROnPlateau` callback to lower the learning rate when validation loss stopped improving.
*   ** Gradient Clipping:** We utilized both `clipvalue` and `clipnorm` in our optimizer to cap the gradients, preventing them from becoming too large during backpropagation.

## How to Run the Project

### 1. Environment Setup
This project requires a specific Python environment with TensorFlow and Keras.

**Prerequisites:**
*   Anaconda or Miniconda installed.
*   Udacity Self-Driving Car Simulator.

**Installation:**
```bash
# Clone the repository
git clone https://github.com/Harkit2004/CVI620_Final_Project.git
cd CVI620_Final_Project

# Create the environment from the yaml file
conda env create -f environment.yaml

# Activate the environment
conda activate cvi620-project
```

### 2. Training the Model
To train the model or explore the data analysis process:
1.  Open the Jupyter Notebook:
    ```bash
    jupyter notebook CVI620_Final_Project.ipynb
    ```
2.  Run the cells to preprocess data, define the model, and start training.
3.  The trained model will be saved as `model.h5`.

### 3. Testing / Autonomous Mode
To see the car drive autonomously:
1.  **Start the Server:**
    ```bash
    python TestSimulation.py
    ```
2.  **Launch the Simulator:**
    *   Open the Udacity Simulator.
    *   Select the track.
    *   Click on **Autonomous Mode**.

The car should now navigate the track automatically!

---
*Project for CVI620 - Computer Vision*