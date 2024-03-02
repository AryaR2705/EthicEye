# DPBH


Our DARK PATTERN project from DARK PATTERN BUSTER HACKATHON

EthicEye a extension which find dark patterns such as 

Forced Action,
Misdirection,
Not Dark Pattern,
Obstruction,
Scarcity,
Sneaking,
Social Proof,
Urgency

for you to run this model you can either use API from model hosted on hugging face : 

import requests

API_URL = "https://api-inference.huggingface.co/models/Arya20032705/dark_pattern"
headers = {"Authorization": "Bearer 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"}

def query(payload):
	response = requests.post(API_URL, headers=headers, json=payload)
	return response.json()
	
output = query({
	"inputs": "20% off buy now fast",
})


OR 


you can download the model from : https://huggingface.co/Arya20032705/dark_pattern

you should then make a folder named as fine_tuned_bert_model
in that folder paste the downloads





