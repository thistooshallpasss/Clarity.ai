---
title: AI Image Upscaler
emoji: ✨
colorFrom: indigo
colorTo: blue
sdk: docker
app_file: app.py
app_port: 5000
---



# Clarity.ai: AI Super-Resolution Web App ✨

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/BlackJack84/ai-image-upscaler)

A complete end-to-end web application that uses a Super-Resolution Generative Adversarial Network (SRGAN) to upscale low-resolution images by 4x. This project demonstrates the entire lifecycle of a deep learning application, from training in PyTorch to deployment via a Flask API.

![data/Landing_page.png](...) ---

## 🚀 About The Project

The goal of this project was to build and understand a complete end-to-end deep learning pipeline. The core of the application is an SRGAN model trained on the high-quality DIV2K dataset. The trained model is served through a Python Flask backend and is accessible via a simple, user-friendly frontend built with HTML, CSS, and JavaScript.

### Built With
* **Backend:** Python, Flask
* **ML/DL:** PyTorch
* **Frontend:** HTML, CSS, JavaScript
* **Dataset:** DIV2K

---

## 🔧 How It Works

The application is powered by a Generative Adversarial Network (GAN) with two main components:
* **Generator:** A deep neural network with 16 residual blocks that takes a low-resolution image and intelligently adds realistic details to upscale it.
* **Discriminator:** A convolutional neural network trained to distinguish between real high-resolution images and the AI-generated ones.

The training process uses a combination of three loss functions to achieve photorealistic results:
1.  **Content Loss (VGG19 Perceptual Loss):** Ensures that the upscaled image has similar features to the original high-resolution image.
2.  **Adversarial Loss:** Pushes the Generator to create images that are indistinguishable from real photos.
3.  **Pixel Loss (L1 Loss):** Minimizes the pixel-wise difference between the generated and real images.

---

## 📊 Results and Analysis

The model was trained for **25 epochs** on the DIV2K training dataset using a Google Colab T4 GPU. The performance was evaluated on the unseen DIV2K validation set.

### Quantitative Results
* **Average Peak Signal-to-Noise Ratio (PSNR): 9.88 dB**
* **Average Structural Similarity Index (SSIM): 0.4263**

### Analysis
These initial scores indicate that the model has successfully learned the fundamental task of upscaling and adding detail. The fluctuating discriminator loss during training showed a healthy adversarial dynamic.

However, the scores are at a baseline level. The primary reason for this is the **limited training duration (25 epochs)**. State-of-the-art SRGAN models are typically trained for hundreds of thousands of iterations, which would significantly improve these metrics. This project serves as a strong proof of concept and a baseline for further training.

### Visual Results
Here is a sample comparison from the validation set after 25 epochs:

![data/Landing_page.png](...) *(Left: Low-Res Input, Right: AI Upscaled Output)*


---

## 🧠 Challenges and Learnings

* **Hardware Limitations:** Training GANs is computationally expensive. This project highlighted the difference in training time and capability between a free-tier GPU and production-level hardware.
* **Large File Management:** The trained model file exceeded GitHub's 100MB limit. This was solved by implementing **Git LFS (Large File Storage)**, which is a standard practice for handling large model artifacts.
* **GAN Dynamics:** Observed the classic "cat-and-mouse" game between the Generator and Discriminator through the fluctuating loss values, providing practical insight into GAN training.

---

## 📈 Future Improvements

* **Extended Training:** Train the model for over 100,000 iterations to significantly improve PSNR/SSIM scores and visual quality.
* **Hyperparameter Tuning:** Experiment with different learning rates, loss function weights, and optimizer settings.
* **Advanced Architecture:** Implement more advanced models like ESRGAN for potentially better results.
* **Deployment:** Deploy the final application to a cloud service like **Hugging Face Spaces** or **Google Cloud Run** for public access.

---

## 🚀 How to Run Locally

To get a local copy up and running, follow these simple steps.

### Prerequisites
* Python 3.8+
* Git and Git LFS installed

### Installation & Setup
1.  Clone the repo:
    ```sh
    git clone https://github.com/thistooshallpasss/Clarity.ai.git
    ```
2.  Navigate to the project directory:
    ```sh
    cd SRGAN_WebApp
    ```
3.  Set up a Python virtual environment:
    ```sh
    python3 -m venv venv
    source venv/bin/activate
    ```
4.  Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```
5.  Download the large model file from Git LFS:
    ```sh
    git lfs pull
    ```
6.  Run the Flask app:
    ```sh
    python app.py
    ```
7.  Open your browser and go to `http://127.0.0.1:5000` 


## ✅ Deployment

The application is successfully deployed and live on **Hugging Face Spaces**. You can access the live demo using the badge at the top of this page.
