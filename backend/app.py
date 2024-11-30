from flask import Flask, request, jsonify, Response, stream_with_context
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import json
from transformers import BertTokenizer, TFBertForSequenceClassification
import numpy as np
import pandas as pd
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Load the fine-tuned BERT model
model_path = "fine_tuned_bert_model"
try:
    loaded_model_bert = TFBertForSequenceClassification.from_pretrained(model_path)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    raise

# Original labels
original_labels = ["Forced Action", "Misdirection", "Not Dark Pattern", "Obstruction", "Scarcity", "Sneaking",
                  "Social Proof", "Urgency"]

def setup_chrome_driver():
    """Setup and return ChromeDriver with appropriate options"""
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--disable-software-rasterizer")
    chrome_options.add_argument("--max_old_space_size=4096")
    
    try:
        # Automatically download and setup ChromeDriver
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=chrome_options)
        driver.set_page_load_timeout(30)
        return driver
    except Exception as e:
        logger.error(f"Error setting up ChromeDriver: {str(e)}")
        raise

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({"error": "No text provided"}), 400
        
        text = data['text']
        prediction = predict_text(text)
        return jsonify(prediction)
    except Exception as e:
        logger.error(f"Error in predict endpoint: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/scrape', methods=['POST'])
def scrape():
    data = request.json
    link = data.get('link')

    if not link:
        return jsonify({"status": "error", "message": "Missing 'link' parameter in request"}), 400

    def generate_output():
        driver = None
        try:
            driver = setup_chrome_driver()
            logger.info(f"Scraping URL: {link}")
            
            # Navigate to the page
            driver.get(link)
            
            # Wait for the page to be fully loaded
            wait = WebDriverWait(driver, 10)
            wait.until(EC.presence_of_element_located((By.TAG_NAME, 'body')))
            
            # Parse page content
            soup = BeautifulSoup(driver.page_source, 'html.parser')
            body_text = soup.get_text(separator='\n', strip=True)
            lines = body_text.split('\n')
            
            # Process text and collect statistics
            line_count = 0
            label_counts = {label: 0 for label in original_labels}
            
            for line in lines:
                if line.strip():  # Only process non-empty lines
                    line_count += 1
                    prediction = predict_text(line)
                    for label in prediction['predicted_labels']:
                        label_counts[label] += 1
            
            # Calculate and yield results
            if line_count > 0:
                label_frequencies = {label: count/line_count for label, count in label_counts.items()}
                yield pd.DataFrame(label_frequencies, index=['Frequency']).to_json(orient='index')
            else:
                yield json.dumps({"status": "error", "message": "No valid text content found"})
                
        except Exception as e:
            logger.error(f"Error during scraping: {str(e)}")
            yield json.dumps({"status": "error", "message": str(e)})
        finally:
            if driver:
                driver.quit()

    return Response(stream_with_context(generate_output()), mimetype='application/json')

def predict_text(text):
    try:
        max_length = 128
        tokens = tokenizer(
            text,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors='tf',
            return_token_type_ids=False,
            return_attention_mask=True
        )
        
        predictions = loaded_model_bert.predict({
            "input_ids": tokens["input_ids"],
            "attention_mask": tokens["attention_mask"]
        })
        
        probabilities = np.exp(predictions.logits) / np.exp(predictions.logits).sum(axis=1, keepdims=True)
        predicted_labels = np.argmax(probabilities, axis=1)
        predicted_labels_original = [original_labels[label] for label in predicted_labels]
        
        return {
            "text": text,
            "predicted_labels": predicted_labels_original,
            "probabilities": probabilities.tolist()
        }
    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}")
        raise

if __name__ == '__main__':
    app.run(debug=True, port=8000)
