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

<img width="658" alt="Screenshot 2024-04-08 at 12 04 45 PM" src="https://github.com/AryaR2705/EthicEye/assets/139691040/dee40121-c01f-4328-b23d-01ce6a9a4869">





Deployed model for inference : https://huggingface.co/Arya20032705/dark_pattern



