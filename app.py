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
import re

st.set_page_config(
    page_title="Multilingual Handwritten Text Reader",
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
    reader = easyocr.Reader(['en', 'ru'], gpu=False)
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

def detect_language(text):
    cyrillic_pattern = re.compile(r'[а-яё]', re.IGNORECASE)
    latin_pattern = re.compile(r'[a-z]', re.IGNORECASE)
    
    cyrillic_count = len(cyrillic_pattern.findall(text))
    latin_count = len(latin_pattern.findall(text))
    
    if cyrillic_count > latin_count:
        return 'cyrillic'
    elif latin_count > 0:
        return 'latin'
    else:
        return 'unknown'

def has_cyrillic(text):
    return bool(re.search(r'[а-яё]', text, re.IGNORECASE))

def correct_word(word, spell_en, russian_words, uzbek_words, language_hint='unknown'):
    if not word:
        return word
    
    lw = word.lower()
    
    if lw in spell_en or lw in russian_words or lw in uzbek_words:
        return word
    
    if language_hint == 'cyrillic' or has_cyrillic(word):
        return word
    elif language_hint == 'latin':
        suggestion = spell_en.correction(word)
        return suggestion if suggestion else word
    else:
        if any(c.isalpha() and ord(c) < 128 for c in word):
            suggestion = spell_en.correction(word)
            return suggestion if suggestion else word
        else:
            return word

def preprocess_and_crop(image, bbox):
    x_coords = [point[0] for point in bbox]
    y_coords = [point[1] for point in bbox]
    
    x_min, x_max = int(min(x_coords)), int(max(x_coords))
    y_min, y_max = int(min(y_coords)), int(max(y_coords))
    
    padding = 8
    h, w = image.shape[:2]
    x_min = max(0, x_min - padding)
    y_min = max(0, y_min - padding)
    x_max = min(w, x_max + padding)
    y_max = min(h, y_max + padding)
    
    cropped = image[y_min:y_max, x_min:x_max]
    
    # Convert to grayscale for better OCR
    if len(cropped.shape) == 3:
        cropped_gray = cv2.cvtColor(cropped, cv2.COLOR_RGB2GRAY)
        # Apply adaptive thresholding to improve text clarity
        cropped_thresh = cv2.adaptiveThreshold(cropped_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        # Convert back to RGB
        cropped = cv2.cvtColor(cropped_thresh, cv2.COLOR_GRAY2RGB)
    
    cropped_pil = Image.fromarray(cropped)
    
    width, height = cropped_pil.size
    if width < 100 or height < 40:
        scale_factor = max(100/width, 40/height, 2.0)
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        cropped_pil = cropped_pil.resize((new_width, new_height), Image.LANCZOS)
    
    return cropped_pil

def choose_best_text(easyocr_text, trocr_text, confidence):
    if not trocr_text.strip():
        return easyocr_text
    
    if not easyocr_text.strip():
        return trocr_text
    
    if has_cyrillic(easyocr_text):
        return easyocr_text
    
    if confidence > 0.8:
        return easyocr_text
    
    if len(trocr_text.strip()) > len(easyocr_text.strip()) * 1.5:
        return trocr_text
    
    return easyocr_text

def main():
    st.title("Multilingual Handwritten Text Reader")
    st.write("Upload an image with handwritten text to extract and recognize it (supports English and Russian).")
    
    with st.sidebar:
        st.header("Settings")
        enable_correction = st.checkbox("Enable Spell Correction", value=True)
        confidence_threshold = st.slider("Detection Confidence Threshold", 0.1, 1.0, 0.4, 0.05)
        paragraph_mode = st.checkbox("Paragraph Mode (merge nearby text)", value=True)
        width_ths = st.slider("Text Width Threshold", 0.1, 2.0, 0.7, 0.1)
        height_ths = st.slider("Text Height Threshold", 0.1, 2.0, 0.7, 0.1)
        use_trocr = st.checkbox("Use TrOCR for English text", value=True)
        
        st.header("Advanced Settings")
        st.write("• **Paragraph Mode**: Groups nearby text together")
        st.write("• **Width/Height Threshold**: Controls text grouping sensitivity")
        st.write("• **Lower values**: More aggressive grouping")
        st.write("• **Higher values**: Less grouping, more individual words")
        
        st.header("About")
        st.write("This app uses EasyOCR for multilingual detection and TrOCR for English handwritten text recognition.")
    
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
            
            try:
                detections = reader.readtext(image_np, paragraph=paragraph_mode, width_ths=width_ths, height_ths=height_ths)
            except Exception as e:
                st.error(f"Error during text detection: {str(e)}")
                detections = reader.readtext(image_np)
            
            if not detections:
                st.warning("No text detected in the image.")
                progress_bar.empty()
                status_text.empty()
                return
            
            filtered_detections = []
            for detection in detections:
                try:
                    if len(detection) >= 3:
                        bbox, text, conf = detection[0], detection[1], detection[2]
                        if conf >= confidence_threshold:
                            filtered_detections.append((bbox, text, conf))
                except Exception as e:
                    continue
            
            if not filtered_detections:
                st.warning("No text detected. Try lowering the confidence threshold.")
                progress_bar.empty()
                status_text.empty()
                return
            
            progress_bar.progress(40)
            status_text.text("Creating visualization...")
            
            vis_image = image_np.copy()
            for bbox, text, conf in filtered_detections:
                try:
                    points = np.array(bbox, np.int32).reshape((-1, 1, 2))
                    color = (0, 255, 0) if has_cyrillic(text) else (255, 0, 0)
                    cv2.polylines(vis_image, [points], True, color, 2)
                except Exception as e:
                    continue
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Detected Regions")
                st.image(vis_image, caption="Green: Cyrillic text, Red: Latin text")
            
            progress_bar.progress(60)
            status_text.text("Processing detected text...")
            
            final_lines = []
            total_regions = len(filtered_detections)
            
            for i, (bbox, easyocr_text, conf) in enumerate(filtered_detections):
                progress_value = 60 + (i / total_regions) * 30
                progress_bar.progress(int(progress_value))
                status_text.text(f"Processing region {i+1}/{total_regions}...")
                
                language_hint = detect_language(easyocr_text)
                
                if use_trocr and not has_cyrillic(easyocr_text) and conf < 0.9:
                    try:
                        cropped = preprocess_and_crop(image_np, bbox)
                        pixel_values = processor(images=cropped, return_tensors="pt").pixel_values
                        
                        with torch.no_grad():
                            generated_ids = model.generate(pixel_values)
                            trocr_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                        
                        final_text = choose_best_text(easyocr_text, trocr_text, conf)
                    except:
                        final_text = easyocr_text
                else:
                    final_text = easyocr_text
                
                if enable_correction and final_text.strip():
                    words = final_text.split()
                    corrected_words = [correct_word(word, spell_en, russian_words, uzbek_words, language_hint) for word in words]
                    final_text = ' '.join(corrected_words)
                
                if final_text.strip():
                    final_lines.append(final_text.strip())
            
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
                    
                    st.subheader("Language Statistics")
                    cyrillic_regions = sum(1 for line in final_lines if has_cyrillic(line))
                    latin_regions = len(final_lines) - cyrillic_regions
                    
                    col_stats1, col_stats2 = st.columns(2)
                    with col_stats1:
                        st.metric("Cyrillic Regions", cyrillic_regions)
                    with col_stats2:
                        st.metric("Latin Regions", latin_regions)
                else:
                    st.warning("No text could be extracted from the image.")

if __name__ == "__main__":
    main()
