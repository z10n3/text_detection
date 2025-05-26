Handwritten Text Reader is a Streamlit-based web application that allows you to detect and recognize handwritten text from images. It uses EasyOCR for text detection and Microsoft's TrOCR model for high-quality handwritten text recognition. Optional spell correction is also supported for English text, with the ability to add Russian and Uzbek wordlists for improved accuracy.


   Features
 Upload image files (.png, .jpg, .jpeg)

 Detect text regions using EasyOCR

 Recognize handwritten text using TrOCR (Microsoft Transformer OCR)

 Optional spell correction (English, custom Russian/Uzbek wordlists)

 Visualize detected text regions on the image

 Download the extracted text as a .txt file

 Adjustable confidence threshold for filtering OCR results

 
  Technologies Used
Streamlit – for building the user interface

EasyOCR – for detecting text regions

TrOCR – for handwritten text recognition

PyTorch – for model inference

Transformers – for TrOCR model

SpellChecker – for English spell correction


  Folder Structure
text-detection/
│
├── app.py                 
├── russian_words.txt       
├── uzbek_words.txt 
├── packages.txt
├── requirements.txt
└── README.md

And there is presentation of this project   
