**__VISIO TrOCR-Based OCR System__**

**Project Overview**
* VISIO is a deep learning based Optical Character Recognition (OCR) system built using Microsoft’s TrOCR models.
* It supports recognition of:
  * Printed text
  * Handwritten text
  * Curved text (using a fine-tuned TrOCR model)
* The system provides accurate text extraction through a Streamlit-based web interface along with text-to-speech audio output.


**Features:**
* TrOCR for printed text recognition
* TrOCR for handwritten text recognition
* Fine-tuned TrOCR model for curved text
* Cleaned and post-processed OCR output
* Text-to-Speech (TTS) for recognized text
* Simple and interactive Streamlit UI
* Fully deep-learning based OCR (no classical OCR engines used)


**Technologies Used:**
* Python
* Streamlit
* PyTorch
* Hugging Face Transformers
* Microsoft TrOCR
* gTTS (Text-to-Speech)





**How to Run the Project (Local Execution):**
+ This project is intended to be executed locally due to the large size of deep learning models and hardware requirements.

Step 1: Clone the Repository
git clone https://github.com/Rijil7/ASAP_TrOCR_Project.git
cd ASAP_TrOCR_Project

Step 2: Create Virtual Environment
python -m venv venv
venv\Scripts\activate   # For Windows

Step 3: Install Requirements
pip install -r requirements.txt

Step 4: Run the Streamlit App
streamlit run streamlit_app.py

The application will open in browser at:
👉 http://localhost:8501


**NOTE:**
* This project uses deep learning OCR models (TrOCR) which are large in size.

* Hence, models are not uploaded directly to this GitHub repository.

* The fine-tuned curved text model is loaded locally or from Hugging Face.

* Internet connection is required for first-time model download.

**OCR Type and Models used:**
+ Printed      ----->     Text	microsoft/trocr-large-printed
+ Handwritten  ----->     Text	microsoft/trocr-large-handwritten
+ Curved Text  ----->     Fine-tuned TrOCR Model
     + The curved text model is a fine-tuned version of TrOCR, trained to recognize distorted and curved text patterns more accurately.
     + Fine-tuned curved text model available at:
https://huggingface.co/rijilraj77/trocr-curved-finetuned
