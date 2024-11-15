from flask import Flask, request, jsonify
import requests
from PIL import Image
from io import BytesIO
from transformers import BartForConditionalGeneration
from transformers import BartTokenizer
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch


from flask import Flask
from flask_cors import CORS

app = Flask(__name__)
CORS(app)


model_directory = "bart-tweet-predictor"
tokenizer = BartTokenizer.from_pretrained(model_directory)
model = BartForConditionalGeneration.from_pretrained(model_directory,use_safetensors=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model_cap = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

def load_image_from_url(url):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    return img

def caption(image):
    inputs = processor(images=image, return_tensors="pt")
    outputs = model_cap.generate(**inputs)
    caption = processor.decode(outputs[0], skip_special_tokens=True)
    return caption

def row_to_json_for_fine_tuning(date,username,company,caption,likes):
    system_prompt = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature."
    user_msg_1 = f"On {date}, {username} from {company} tweeted with a media which depicts {caption}. As someone well-versed in Twitter trends and engagement, create a tweet that resonates well with the audience and aligns with the style and tone of the original tweet."
    return f"[INST] <>\n{{ {system_prompt} }}\n<>\n\n{{ {user_msg_1} }} [/INST]"

def generate_prediction(example):
    inputs = tokenizer(example, return_tensors="pt", padding=True, truncation=True)#.to(device)
    outputs = model.generate(
    **inputs,
    max_length=50,
    do_sample=True,
    top_k=50,
    top_p=0.95,
    temperature=0.8,
    pad_token_id=tokenizer.eos_token_id,
    eos_token_id=tokenizer.eos_token_id
    )
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    username = data['username']
    datetime = data['datetime']
    likes = data['likes']
    media = data['media']
    company = data['company']

    img = load_image_from_url(media)
    cap = caption(img)

    input_data = row_to_json_for_fine_tuning(datetime,username,company,cap,likes)
    
    prediction = generate_prediction(input_data)
    
    return jsonify({'predicted_content': prediction})

url = "https://pbs.twimg.com/media/Eo8N3JLVoAAlDJT?format=jpg&name=small"
username = "timhortonsph"
company = "tim hortons"
likes = 1
img =  load_image_from_url(url)
cap = caption(img)
input_data = row_to_json_for_fine_tuning("",username,company,cap,likes)
prediction = generate_prediction(input_data)
print(prediction)

if __name__ == '__main__':
    app.run(debug=True)