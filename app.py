import streamlit as st
from PIL import Image
import torch
import numpy as np
import cv2
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from spellchecker import SpellChecker
import easyocr
import time
import os

st.set_page_config(
    page_title="Handwritten Text Reader",
    page_icon="✏️",
    layout="centered"
)

st.markdown("""
<style>
    .main { padding: 2rem; }
    h1 { color: #1E3A8A; }
    .stButton button { background-color: #1E3A8A; color: white; }
    .result-container { background-color: #f8f9fa; border-radius: 5px; padding: 20px; margin-top: 20px; border: 1px solid #ddd; }
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
        return set()
    
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return set(word.strip().lower() for word in f.readlines() if word.strip())
    except UnicodeDecodeError:
        try:
            with open(path, 'r', encoding='cp1251') as f:
                return set(word.strip().lower() for word in f.readlines() if word.strip())
        except:
            return set()
    except:
        return set()

def correct_word(word, spell_en, russian_words, uzbek_words):
    if not word:
        return word
    
    lw = word.lower()
    if lw in spell_en:
        return word
    elif lw in russian_words or lw in uzbek_words:
        return word
    else:
        suggestion = spell_en.correction(word)
        return suggestion if suggestion else word

def preprocess_and_crop(image, bbox):
    x_coords = [point[0] for point in bbox]
    y_coords = [point[1] for point in bbox]
    
    x_min, x_max = int(min(x_coords)), int(max(x_coords))
    y_min, y_max = int(min(y_coords)), int(max(y_coords))
    
    padding = 2
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
        st.header("Settings")
        enable_correction = st.checkbox("Enable Spell Correction", value=True)
        confidence_threshold = st.slider("Detection Confidence Threshold", 0.1, 1.0, 0.4, 0.05)
        
        st.header("About")
        st.write("This app uses EasyOCR for text detection and TrOCR for handwritten text recognition.")
    
    processor, model, reader = load_models()
    
    russian_words = load_wordlist("russian_words.txt")
    uzbek_words = load_wordlist("uzbek_words.txt")
    spell_en = SpellChecker()
    
    uploaded_file = st.file_uploader("Choose an image file", type=["png", "jpg", "jpeg"])
    
    if uploaded_file is not None:
        image_pil = Image.open(uploaded_file).convert("RGB")
        st.image(image_pil, caption="Uploaded Image", use_container_width=True)
        
        if st.button("Extract Text", type="primary"):
            image_np = np.array(image_pil)
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.text("Detecting text regions...")
            progress_bar.progress(20)
            
            detections = reader.readtext(image_np)
            
            filtered_detections = [
                (bbox, text, conf) 
                for bbox, text, conf in detections 
                if conf >= confidence_threshold
            ]
            
            if not filtered_detections:
                st.warning("No text detected. Try lowering the confidence threshold.")
                progress_bar.empty()
                status_text.empty()
                return
            
            progress_bar.progress(40)
            status_text.text("Creating visualization...")
            
            vis_image = image_np.copy()
            for bbox, _, conf in filtered_detections:
                points = np.array(bbox, np.int32).reshape((-1, 1, 2))
                cv2.polylines(vis_image, [points], True, (0, 255, 0), 2)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Detected Regions")
                st.image(vis_image, caption="Green boxes show detected text regions")
            
            progress_bar.progress(60)
            status_text.text("Processing with TrOCR...")
            
            final_lines = []
            total_regions = len(filtered_detections)
            
            for i, (bbox, easyocr_text, conf) in enumerate(filtered_detections):
                progress_value = 60 + (i / total_regions) * 30
                progress_bar.progress(int(progress_value))
                status_text.text(f"Processing region {i+1}/{total_regions}...")
                
                cropped = preprocess_and_crop(image_np, bbox)
                
                try:
                    pixel_values = processor(images=cropped, return_tensors="pt").pixel_values
                    
                    with torch.no_grad():
                        generated_ids = model.generate(pixel_values)
                        trocr_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                    
                    if enable_correction and trocr_text.strip():
                        words = trocr_text.split()
                        corrected_words = [correct_word(word, spell_en, russian_words, uzbek_words) for word in words]
                        final_text = ' '.join(corrected_words)
                    else:
                        final_text = trocr_text
                    
                    if final_text.strip():
                        final_lines.append(final_text.strip())
                        
                except:
                    if easyocr_text.strip():
                        final_lines.append(easyocr_text.strip())
            
            progress_bar.progress(100)
            status_text.text("Processing complete!")
            time.sleep(1)
            progress_bar.empty()
            status_text.empty()
            
            with col2:
                st.subheader("Extracted Text")
                
                if final_lines:
                    final_text = '\n'.join(final_lines)
                    st.markdown(f'<div class="result-container"><pre>{final_text}</pre></div>', 
                               unsafe_allow_html=True)
                    
                    st.download_button(
                        label="Download Text",
                        data=final_text,
                        file_name="extracted_text.txt",
                        mime="text/plain"
                    )
                else:
                    st.warning("No text could be extracted from the image.")

if __name__ == "__main__":
    main()