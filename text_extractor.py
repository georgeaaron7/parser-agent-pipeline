from sentence_transformers import SentenceTransformer
import pytesseract
from PIL import Image
import cv2
import numpy as np
import logging
import json
import uuid
from pathlib import Path
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TextExtractor:
    def close(self):
        """No vector DB connection to close for FAISS"""
        pass

    def __init__(self):
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
    def extract_text_from_image(self, image_path):
        """Extract text from a single image using Tesseract OCR"""
        try:
            img = Image.open(image_path)
            img = img.convert("RGB")
            text = pytesseract.image_to_string(img)
            return text.strip()
        except Exception as e:
            logger.error(f"Error extracting text from image {image_path}: {e}")
            return ""

    def extract_text_from_pdf(self, pdf_path):
        """Extract text from a PDF file (multi-page)"""
        all_text = []
        try:
            pdf = cv2.PDF(pdf_path)
            for page_number in range(len(pdf)):
                text = pytesseract.image_to_string(pdf[page_number])
                all_text.append({"page_number": page_number + 1, "text": text})
                
        except Exception as e:
            logger.error(f"Error extracting text from PDF {pdf_path}: {e}")
        
        return all_text
    
    def chunk_text(self, text, max_length=500):
        """Chunk long text into smaller parts"""
        chunks = []
        words = text.split()
        current_chunk = []
        current_length = 0
        
        for word in words:
            word_length = len(word) + 1  
            if current_length + word_length > max_length:
                chunks.append(" ".join(current_chunk))
                current_chunk = [word]
                current_length = word_length
            else:
                current_chunk.append(word)
                current_length += word_length
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks
    
    def process_and_store(self, source_path, document_id, is_image=True):
        """Process an image or PDF file and store the extracted text in chunks"""
        try:
            if is_image:
                text = self.extract_text_from_image(source_path)
                logger.info(f"Extracted text from image: {text[:100]}...")  
            else:
                all_text = self.extract_text_from_pdf(source_path)
                logger.info(f"Extracted text from PDF with {len(all_text)} pages")
                for page in all_text:
                    logger.info(f"Page {page['page_number']} text: {page['text'][:100]}...")  
            
        except Exception as e:
            logger.error(f"Error processing and storing data from {source_path}: {e}")
            # logger.info(f"Stored {chunk_counter} text chunks from page {page_number}")
            return True
            
        except Exception as e:
            logger.error(f"Error storing text in Weaviate: {e}")
            return False
