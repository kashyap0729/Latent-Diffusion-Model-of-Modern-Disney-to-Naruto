#  Fine-Tuning a Latent Diffusion Model of Modern Disney LLM with Naruto-Style Image Generation

This project focuses on fine-tuning **Stable Diffusion v1.5**—a powerful Latent Diffusion Model (LDM)—to generate images in the style of the **Naruto** anime, based on custom text prompts. The model is fine-tuned on a stylized dataset, and only the **U-Net** component is updated during training, while the **CLIP tokenizer** and **VAE encoder-decoder** remain frozen to maintain efficiency. A **Streamlit** app is also developed to offer an intuitive interface for generating Naruto-style images from textual descriptions.

---

## 🎯 Project Objective

To create a lightweight and efficient **text-to-image generation model** that produces **Naruto-inspired images**, using minimal resources and fine-tuning only the U-Net part of a pre-trained latent diffusion model.

---

## 📁 Dataset Overview

* **Dataset Name**: [Naruto BLIP Captions Dataset](https://huggingface.co/datasets/lambdalabs/naruto-blip-captions)
* **Provider**: Lambda Labs (via Hugging Face)
* **Image Count**: Over 1,200 stylized JPEG images depicting Naruto anime-style characters and scenes.
* **Captions**: Each image is paired with a caption generated using **BLIP (Bootstrapping Language-Image Pretraining)**, creating stylized textual prompts suitable for training.
* **Total Size**: Approximately 700 MB
* **Usage**: Used exclusively for training purposes (no test/validation split)

---

## 🧠 Model Components & Architecture

The fine-tuning approach leverages **Stable Diffusion v1.5**, composed of the following modules:

* **CLIP Text Encoder**
  Transforms input text prompts into dense embeddings used to guide image generation. This component remains **frozen** during training.

* **Variational Autoencoder (VAE)**
  Converts images into compressed latent representations and reconstructs them post-generation. Both the **encoder and decoder** parts of the VAE are **kept frozen**.

* **U-Net Denoising Network**
  The core of the model that is **fine-tuned**. It learns how to progressively denoise latent representations to generate Naruto-style images aligned with the prompt.

* **Latent Diffusion Mechanism**
  Enables efficient image synthesis by working in a compressed latent space instead of full-resolution images.

---

## 🔧 Training Configuration

* **Base Model**: Stable Diffusion v1.5
* **Scope of Fine-Tuning**: Only the **U-Net** module is updated; CLIP and VAE remain unchanged.
* **Epochs**:

  * Standard training: **10 epochs**
  * Modified Diffusion variant (**Mo-Di-Diffusion**): **5 epochs**
* **Frameworks**:

  * **PyTorch**
  * **Hugging Face Diffusers & Transformers**

This setup allows for efficient training while preserving the core knowledge of the base model.

---

## 🖼️ Streamlit Web App

An easy-to-use **Streamlit-based application** was built for real-time image generation. It allows users to:

* Enter custom text prompts (e.g., “Naruto fighting in the rain”)
* Generate and view Naruto-style images
* Experience a smooth and interactive UI designed for both demo and testing use cases

The app is built entirely in Python and connects the frontend prompt input with the backend image generation pipeline.

---

## 📊 Model Evaluation: Pre vs. Post Fine-Tuning

<img width="646" height="345" alt="image" src="https://github.com/user-attachments/assets/993dc526-3b11-4e09-aaa4-57392cd01dd7" />

| **Metric**                           | **Before Fine-Tuning** | **After Fine-Tuning** | **Change** | **Remarks**                                                         |
| ------------------------------------ | ---------------------- | --------------------- | ---------- | ------------------------------------------------------------------- |
| **CLIP Score**                       | 23.55                  | 24.76                 | +1.21      | Indicates **better alignment** between prompts and generated images |
| **LPIPS (Perceptual Similarity)**    | 0.781                  | 0.709                 | -0.072     | Lower value suggests improved **visual similarity** to Naruto style |
| **FID (Fréchet Inception Distance)** | -                      | 412.53                | -          | High value; still **room to improve** overall image fidelity        |
| **Inception Score**                  | nan                    | nan                   | -          | **Not computed** due to instability in output image classes         |

---

## 📸 Sample Result

Post fine-tuning, the model produces visually enhanced Naruto-style images. Below are examples showcasing improvements in both character fidelity and alignment with text prompts:

<img width="656" height="368" alt="image" src="https://github.com/user-attachments/assets/b774fdcd-552b-47f0-b0b5-120653b6b1d6" />

These images illustrate that the model has successfully learned Naruto-specific features such as character outfits, hairstyles, and facial structures, while preserving general prompt alignment.

---

## 🚀 How to Get Started

Here’s a step-by-step guide for running the project:

1. **Clone the Repository**

   ```bash
   git clone https://github.com/your-username/LDM-Finetune-Naruto.git
   cd LDM-Finetune-Naruto
   ```

2. **Open Google Colab**

   * Open the Jupyter/Colab notebook provided in the repo
   * Ensure you have access to a GPU with at least **25 GB of VRAM** (A100 recommended)

3. **Install Dependencies**

   * The notebook will handle installation for required libraries like `transformers`, `diffusers`, `streamlit`, etc.

4. **Run the Fine-Tuning Script**

   * Execute the training notebook cells step-by-step
   * Training U-Net will take a few hours depending on hardware

5. **Launch the Streamlit App**

   ```bash
   streamlit run app.py
   ```

   * Open the URL generated to access the UI
   * Input a text prompt and watch your Naruto-style image come to life!

---

## 💡 Future Work

* Enhance image quality by incorporating **LoRA** (Low-Rank Adaptation) or **DreamBooth**
* Use a validation set and better prompt diversity to improve generalization
* Train a **custom scheduler** or explore **textual inversion** for stylized prompt boosting
* Deploy the Streamlit app on **Hugging Face Spaces** or **AWS EC2** for public access

