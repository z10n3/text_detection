import streamlit as st
from PIL import Image
import torch
import numpy as np
import os
import cv2
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from spellchecker import SpellChecker
import easyocr
import time

st.set_page_config(
    page_title="Handwritten Text Reader",
    page_icon="✏️",
    layout="centered",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main {
        padding: 2rem;
    }
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    h1 {
        color: #1E3A8A;
    }
    .stButton button {
        background-color: #1E3A8A;
        color: white;
    }
    .image-container {
        border: 1px solid #ddd;
        border-radius: 5px;
        padding: 10px;
    }
    .result-container {
        background-color: #f8f9fa;
        border-radius: 5px;
        padding: 20px;
        margin-top: 20px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
    model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
    reader = easyocr.Reader(['en'], gpu=False)
    return processor, model, reader

@st.cache_data
def load_wordlist(path):
    if not os.path.exists(path):
        with open(path, 'w', encoding='utf-8') as f:
            pass
    
    with open(path, 'r', encoding='utf-8') as f:
        return set(word.strip().lower() for word in f.readlines() if word.strip())

def correct_word(word, spell_en, russian_words, uzbek_cyrillic_words):
    lw = word.lower()
    if lw in spell_en:
        return word
    elif lw in russian_words or lw in uzbek_cyrillic_words:
        return word
    else:
        suggestion = spell_en.correction(word)
        return suggestion if suggestion else word

def preprocess_and_crop(image, bbox):
    x_min, y_min = map(int, bbox[0])
    x_max, y_max = map(int, bbox[2])
    padding = 5
    h, w = image.shape[:2]
    x_min = max(0, x_min - padding)
    y_min = max(0, y_min - padding)
    x_max = min(w, x_max + padding)
    y_max = min(h, y_max + padding)
    
    cropped = image[y_min:y_max, x_min:x_max]
    return Image.fromarray(cropped)

def main():
    st.title("Handwritten Text Reader")
    st.write("Upload an image with handwritten text to extract and recognize it.")
    
    with st.sidebar:
        st.header("Options")
        enable_correction = st.checkbox("Enable Spell Correction", value=True)
        confidence_threshold = st.slider("Detection Confidence Threshold", 0.1, 1.0, 0.5, 0.05)
        
        st.header("About")
        st.write("""
        This app uses EasyOCR for text detection and TrOCR for handwritten text recognition.
        It supports English text recognition with optional spell correction.
        """)
    
    with st.spinner("Loading models (this may take a moment)..."):
        processor, model, reader = load_models()
    
    russian_words = load_wordlist("russian_words.txt")
    uzbek_cyrillic_words = load_wordlist("uzbek_words.txt")
    spell_en = SpellChecker()
    
    uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
    
    if uploaded_file is not None:
        image_pil = Image.open(uploaded_file).convert("RGB")
        st.markdown('<div class="image-container">', unsafe_allow_html=True)
        st.image(image_pil, caption="Uploaded Image", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        if st.button("Extract Text"):
            image_np = np.array(image_pil)
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.text("Detecting text regions...")
            progress_bar.progress(10)
            
            detections = reader.readtext(image_np, min_size=20, 
                                         paragraph=False, 
                                         decoder='greedy',
                                         beamWidth=5,
                                         batch_size=8,
                                         allowlist=None,
                                         blocklist=None,
                                         detail=1,
                                         rotation_info=None,
                                         paragraph_threshold=0.5,
                                         contrast_ths=0.1,
                                         adjust_contrast=0.5,
                                         text_threshold=0.7,
                                         link_threshold=0.4,
                                         low_text=0.4,
                                         mag_ratio=1.5)
            
            filtered_detections = [det for det in detections if det[2] >= confidence_threshold]
            
            progress_bar.progress(40)
            status_text.text("Processing text with TrOCR...")
            
            col1, col2 = st.columns(2)
            
            vis_image = image_np.copy()
            for bbox, _, conf in filtered_detections:
                points = np.array(bbox, np.int32).reshape((-1, 1, 2))
                cv2.polylines(vis_image, [points], True, (0, 255, 0), 2)
            
            with col1:
                st.subheader("Detected Regions")
                st.image(vis_image, caption="Text Regions", use_container_width=True)
            
            final_lines = []
            
            total_regions = len(filtered_detections)
            for i, (bbox, _, _) in enumerate(filtered_detections):
                
                progress_value = 40 + (i / total_regions) * 50
                progress_bar.progress(int(progress_value))
                status_text.text(f"Processing region {i+1}/{total_regions}...")
        
                cropped = preprocess_and_crop(image_np, bbox)
                

                pixel_values = processor(images=cropped, return_tensors="pt").pixel_values
                with torch.no_grad():
                    generated_ids = model.generate(pixel_values)
                    decoded = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                
                if enable_correction:
                    words = decoded.split()
                    corrected_line = ' '.join(correct_word(word, spell_en, russian_words, uzbek_cyrillic_words) for word in words)
                    final_lines.append(corrected_line)
                else:
                    final_lines.append(decoded)
            
            progress_bar.progress(100)
            status_text.text("Processing complete!")
            time.sleep(1)
            status_text.empty()
            progress_bar.empty()
            
            with col2:
                st.subheader("Recognized Text")
                final_text = '\n'.join(final_lines)
                st.markdown(f'<div class="result-container"><pre>{final_text}</pre></div>', unsafe_allow_html=True)
            
            st.download_button(
                label="Download text",
                data=final_text,
                file_name="extracted_text.txt",
                mime="text/plain"
            )

if __name__ == "__main__":
    main()