import streamlit as st
from PIL import Image, ImageEnhance, ImageFilter
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
    page_title="Handwritten Text Reader (Cyrillic Enhanced)",
    page_icon="✏️",
    layout="centered"
)

st.markdown("""
<style>
    .main { padding: 2rem; }
    h1 { color: #1E3A8A; }
    .stButton button { background-color: #1E3A8A; color: white; }
    .result-container { 
        background-color: #f8f9fa; 
        border-radius: 5px; 
        padding: 20px; 
        margin-top: 20px; 
        border: 1px solid #ddd;
        font-family: 'Courier New', monospace;
        font-size: 14px;
        line-height: 1.5;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    try:
        processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
        model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
        reader = easyocr.Reader(['en', 'ru', 'uk', 'bg'], gpu=False, verbose=False)
        return processor, model, reader
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None, None, None

@st.cache_data
def load_wordlist(path):
    if not os.path.exists(path):
        with open(path, 'w', encoding='utf-8') as f:
            pass
        return set()
    
    encodings_to_try = ['utf-8', 'cp1251', 'utf-16', 'iso-8859-5']
    
    for encoding in encodings_to_try:
        try:
            with open(path, 'r', encoding=encoding) as f:
                words = set()
                for line in f:
                    word = line.strip().lower()
                    if word and len(word) > 1:
                        words.add(word)
                return words
        except (UnicodeDecodeError, UnicodeError):
            continue
        except Exception:
            break
    
    return set()

def detect_language(text):
    cyrillic_pattern = re.compile(r'[а-яё]', re.IGNORECASE)
    return 'cyrillic' if cyrillic_pattern.search(text) else 'latin'

def enhance_image_for_ocr(image):
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()
    
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    
    denoised = cv2.bilateralFilter(enhanced, 9, 75, 75)
    
    _, binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    kernel = np.ones((1,1), np.uint8)
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    return cleaned

def correct_word(word, spell_en, russian_words, uzbek_words):
    if not word or len(word) < 2:
        return word
    
    lw = word.lower()
    lang = detect_language(word)
    
    if lang == 'cyrillic':
        if lw in russian_words or lw in uzbek_words:
            return word
        else:
            for ref_word in russian_words:
                if abs(len(ref_word) - len(lw)) <= 2:
                    similarity = sum(c1 == c2 for c1, c2 in zip(lw, ref_word)) / max(len(lw), len(ref_word))
                    if similarity > 0.7:
                        return ref_word
            return word
    else:
        if lw in spell_en:
            return word
        else:
            suggestion = spell_en.correction(word)
            return suggestion if suggestion and suggestion != word else word

def preprocess_and_crop(image, bbox):
    x_coords = [point[0] for point in bbox]
    y_coords = [point[1] for point in bbox]
    
    x_min, x_max = int(min(x_coords)), int(max(x_coords))
    y_min, y_max = int(min(y_coords)), int(max(y_coords))
    
    padding = 5
    h, w = image.shape[:2]
    x_min = max(0, x_min - padding)
    y_min = max(0, y_min - padding)
    x_max = min(w, x_max + padding)
    y_max = min(h, y_max + padding)
    
    cropped = image[y_min:y_max, x_min:x_max]
    
    if cropped.size == 0:
        return None
    
    enhanced = enhance_image_for_ocr(cropped)
    
    if len(enhanced.shape) == 2:
        enhanced_rgb = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
    else:
        enhanced_rgb = enhanced
    
    return Image.fromarray(enhanced_rgb)

def main():
    st.title("Handwritten Text Reader (Cyrillic Enhanced)")
    st.write("Upload an image with handwritten text to extract and recognize it. Enhanced for Cyrillic text recognition.")
    
    with st.sidebar:
        st.header("Settings")
        enable_correction = st.checkbox("Enable Spell Correction", value=True)
        confidence_threshold = st.slider("Detection Confidence Threshold", 0.1, 1.0, 0.3, 0.05)
        enhance_images = st.checkbox("Enhance Images for OCR", value=True)
        
        st.header("Language Support")
        st.write("• English")
        st.write("• Russian (Русский)")
        st.write("• Ukrainian (Українська)")
        st.write("• Bulgarian (Български)")
        
        st.header("About")
        st.write("This app uses EasyOCR for text detection and TrOCR for handwritten text recognition with enhanced Cyrillic support.")
    
    models = load_models()
    if not all(models):
        st.error("Failed to load required models. Please check your internet connection and try again.")
        return
    
    processor, model, reader = models
    
    russian_words = load_wordlist("russian_words.txt")
    uzbek_words = load_wordlist("uzbek_words.txt")
    spell_en = SpellChecker()
    
    uploaded_file = st.file_uploader("Choose an image file", type=["png", "jpg", "jpeg", "bmp", "tiff"])
    
    if uploaded_file is not None:
        image_pil = Image.open(uploaded_file).convert("RGB")
        st.image(image_pil, caption="Uploaded Image", use_container_width=True)
        
        if st.button("Extract Text", type="primary"):
            image_np = np.array(image_pil)
            
            if enhance_images:
                enhanced_image = enhance_image_for_ocr(image_np)
                detection_image = enhanced_image
            else:
                detection_image = image_np
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.text("Detecting text regions...")
            progress_bar.progress(20)
            
            try:
                detections = reader.readtext(detection_image, paragraph=False, width_ths=0.8, height_ths=0.8)
            except Exception as e:
                st.error(f"Error during text detection: {str(e)}")
                progress_bar.empty()
                status_text.empty()
                return
            
            filtered_detections = [
                (bbox, text, conf) 
                for bbox, text, conf in detections 
                if conf >= confidence_threshold
            ]
            
            if not filtered_detections:
                st.warning("""
                No text detected. Try:
                - Lowering the confidence threshold
                - Using a higher resolution image
                - Ensuring the handwriting is clear and well-lit
                - Trying with image enhancement enabled
                """)
                progress_bar.empty()
                status_text.empty()
                return
            
            progress_bar.progress(40)
            status_text.text("Creating visualization...")
            
            vis_image = image_np.copy()
            for bbox, text, conf in filtered_detections:
                points = np.array(bbox, np.int32).reshape((-1, 1, 2))
                cv2.polylines(vis_image, [points], True, (0, 255, 0), 2)
                
                lang = detect_language(text)
                color = (255, 0, 0) if lang == 'cyrillic' else (0, 255, 0)
                cv2.polylines(vis_image, [points], True, color, 2)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Detected Regions")
                st.image(vis_image, caption="Green: Latin text, Red: Cyrillic text")
            
            progress_bar.progress(60)
            status_text.text("Processing with TrOCR...")
            
            final_lines = []
            total_regions = len(filtered_detections)
            
            for i, (bbox, easyocr_text, conf) in enumerate(filtered_detections):
                progress_value = 60 + (i / total_regions) * 30
                progress_bar.progress(int(progress_value))
                status_text.text(f"Processing region {i+1}/{total_regions}...")
                
                cropped = preprocess_and_crop(detection_image if enhance_images else image_np, bbox)
                
                if cropped is None:
                    continue
                
                final_text = ""
                
                try:
                    pixel_values = processor(images=cropped, return_tensors="pt").pixel_values
                    
                    with torch.no_grad():
                        generated_ids = model.generate(pixel_values, max_length=200)
                        trocr_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                    
                    if trocr_text.strip():
                        final_text = trocr_text.strip()
                    elif easyocr_text.strip():
                        final_text = easyocr_text.strip()
                        
                except Exception as e:
                    if easyocr_text.strip():
                        final_text = easyocr_text.strip()
                
                if enable_correction and final_text:
                    words = final_text.split()
                    corrected_words = []
                    for word in words:
                        cleaned_word = re.sub(r'[^\w\s\u0400-\u04FF]', '', word)
                        if cleaned_word:
                            corrected_word = correct_word(cleaned_word, spell_en, russian_words, uzbek_words)
                            corrected_words.append(corrected_word)
                    if corrected_words:
                        final_text = ' '.join(corrected_words)
                
                if final_text:
                    final_lines.append(final_text)
            
            progress_bar.progress(100)
            status_text.text("Processing complete!")
            time.sleep(1)
            progress_bar.empty()
            status_text.empty()
            
            with col2:
                st.subheader("Extracted Text")
                
                if final_lines:
                    final_text = '\n'.join(final_lines)
                    
                    cyrillic_count = len(re.findall(r'[а-яё]', final_text, re.IGNORECASE))
                    total_chars = len(re.findall(r'[a-zA-Zа-яё]', final_text, re.IGNORECASE))
                    
                    if total_chars > 0:
                        cyrillic_percentage = (cyrillic_count / total_chars) * 100
                        st.info(f"Detected text: {cyrillic_percentage:.1f}% Cyrillic, {100-cyrillic_percentage:.1f}% Latin")
                    
                    st.markdown(f'<div class="result-container">{final_text}</div>', 
                               unsafe_allow_html=True)
                    
                    st.download_button(
                        label="Download Text",
                        data=final_text,
                        file_name="extracted_text.txt",
                        mime="text/plain"
                    )
                    
                    with st.expander("View Individual Detections"):
                        for i, (_, easyocr_text, conf) in enumerate(filtered_detections):
                            lang = detect_language(easyocr_text)
                            st.write(f"**Region {i+1}** ({lang}, confidence: {conf:.2f}): {easyocr_text}")
                else:
                    st.warning("No text could be extracted from the image.")

if __name__ == "__main__":
    main()
