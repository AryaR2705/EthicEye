from flask import Flask, request, jsonify, Response, stream_with_context
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import json
from transformers import BertTokenizer, TFBertForSequenceClassification
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the fine-tuned BERT model
model_path = "fine_tuned_bert_model"  # Change to the path where you saved your trained model
loaded_model_bert = TFBertForSequenceClassification.from_pretrained(model_path)

# Load the tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Assuming you have the original labels used during training
original_labels = ["Forced Action", "Misdirection", "Not Dark Pattern", "Obstruction", "Scarcity", "Sneaking",
                   "Social Proof", "Urgency"]

@app.route('/predict', methods=['POST'])
def predict():
    # Get text from request
    data = request.get_json()
    text = data['text']

    # Tokenize and pad the new text
    max_length = 128  # Adjust as needed
    new_texts_tokens_bert = tokenizer(text, truncation=True, padding=True, max_length=max_length, return_tensors='tf',
                                      return_token_type_ids=False, return_attention_mask=True)

    # Make predictions using the loaded BERT model
    predictions_bert = loaded_model_bert.predict(
        {"input_ids": new_texts_tokens_bert["input_ids"], "attention_mask": new_texts_tokens_bert["attention_mask"]})

    # Get the predicted class probabilities
    probabilities_bert = np.exp(predictions_bert.logits) / np.exp(
        predictions_bert.logits).sum(axis=1, keepdims=True)

    # Get the predicted labels (argmax of probabilities)
    predicted_labels_bert = np.argmax(probabilities_bert, axis=1)

    # Convert predicted labels to original labels
    predicted_labels_original_bert = [original_labels[label] for label in predicted_labels_bert]

    # Construct response
    response = {
        "text": text,
        "predicted_labels": predicted_labels_original_bert,
        "probabilities": probabilities_bert.tolist()
    }

    return jsonify(response)

@app.route('/scrape', methods=['POST'])
def scrape():
    data = request.json
    link = data.get('link')

    if not link:
        return jsonify({"status": "error", "message": "Missing 'link' parameter in request"})

    # Selenium setup
    chrome_options = Options()
    chrome_options.add_argument("--headless")  # Ensure GUI is off
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-gpu")  # Reduce resource usage
    chrome_options.add_argument("--disable-software-rasterizer")  # Additional option for resource optimization
    chrome_options.add_argument("--max_old_space_size=4096")  # Limit memory usage

    def generate_output():
        line_count = 0  # Initialize line count
        label_counts = {label: 0 for label in original_labels}  # Initialize label counts
        try:
            # Initialize WebDriver with a timeout of 30 seconds
            driver = webdriver.Chrome(options=chrome_options)
            driver.set_page_load_timeout(30)

            # Navigate to the page
            app.logger.info(f"Scraping URL: {link}")
            driver.get(link)

            # Wait for the page to be fully loaded
            wait = WebDriverWait(driver, 10)
            wait.until(EC.presence_of_element_located((By.TAG_NAME, 'body')))

            # Use BeautifulSoup to parse the page source
            soup = BeautifulSoup(driver.page_source, 'html.parser')

            # Extract text content from the page
            body_text = soup.get_text(separator='\n', strip=True)
            app.logger.info("Scraped Body Text: [Content too long, not displayed]")

            # Split the text into lines
            lines = body_text.split('\n')

            # Process each line with the BERT model
            for line in lines:
                line_count += 1  # Increment line count
                # Make prediction using the BERT model
                prediction_response = predict_text(line)

                # Update label counts
                for label in prediction_response['predicted_labels']:
                    label_counts[label] += 1

            # Calculate label frequencies
            label_frequencies = {label: count / line_count for label, count in label_counts.items()}

            # Yield label frequencies in tabular format
            yield pd.DataFrame(label_frequencies, index=['Frequency']).to_json(orient='index')

        except Exception as e:
            app.logger.error(f"Error during scraping: {str(e)}")
            yield json.dumps({"status": "error", "message": str(e)}) + '\n'

        finally:
            if 'driver' in locals():
                driver.quit()

    return Response(stream_with_context(generate_output()), mimetype='application/json')

def predict_text(text):
    # Tokenize and pad the text
    max_length = 128  # Adjust as needed
    new_texts_tokens_bert = tokenizer(text, truncation=True, padding=True, max_length=max_length, return_tensors='tf',
                                      return_token_type_ids=False, return_attention_mask=True)

    # Make predictions using the loaded BERT model
    predictions_bert = loaded_model_bert.predict(
        {"input_ids": new_texts_tokens_bert["input_ids"], "attention_mask": new_texts_tokens_bert["attention_mask"]})

    # Get the predicted class probabilities
    probabilities_bert = np.exp(predictions_bert.logits) / np.exp(
        predictions_bert.logits).sum(axis=1, keepdims=True)

    # Get the predicted labels (argmax of probabilities)
    predicted_labels_bert = np.argmax(probabilities_bert, axis=1)

    # Convert predicted labels to original labels
    predicted_labels_original_bert = [original_labels[label] for label in predicted_labels_bert]

    # Construct and return prediction response
    prediction_response = {
        "predicted_labels": predicted_labels_original_bert,
    }

    return prediction_response

if __name__ == '__main__':
    app.run(debug=True, port=8000)
