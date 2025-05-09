from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
from dotenv import load_dotenv
import numpy as np
import requests
import torch
import json
import io
import os
from typing import Optional

# Loading Environment Variable to access the Hugging Face API Token
load_dotenv()
api_key = os.getenv("API_KEY")
HUGGINGFACE_API_KEY = api_key

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensuring static image directory exists for serving product images
os.makedirs("images", exist_ok=True)
app.mount("/images", StaticFiles(directory="images"), name="images")
# app.mount("/", StaticFiles(directory="frontend_build", html=True), name="frontend")


with open("products.json", "r") as f:
    products = json.load(f)

CATEGORY_KEYWORDS = {
    "shoes": ["shoes", "sneaker", "footwear", "boots"],
    "apparel": ["t-shirt", "shirt", "tee", "clothing", "jacket", "pants", "dress"],
    "electronics": ["watch", "smartwatch", "headphones", "headphone", "earbuds", "speaker"],
    "accessories": ["backpack", "bag", "purse", "wallet", "sunglasses"]
}

# Load the sentence transformer model for text similarity
text_model = SentenceTransformer("all-MiniLM-L6-v2")
for p in products:
    p["text_embedding"] = text_model.encode(p["title"] + " " + p["description"]).tolist()

# Initialize CLIP model for image processing
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
for p in products:
    inputs = clip_processor(
        text=p["title"] + " " + p["description"],
        return_tensors="pt",
        padding=True,
        truncation=True
    )
    with torch.no_grad():
        text_features = clip_model.get_text_features(**inputs)
        p["clip_embedding"] = text_features.squeeze().numpy()

class ChatRequest(BaseModel):
    message: str

def extract_category_from_query(query: str) -> Optional[str]:
    query = query.lower()
    for category, keywords in CATEGORY_KEYWORDS.items():
        for keyword in keywords:
            if keyword in query:
                return category
    return None

def predict_category_from_image(image):
    category_prompts = [
        "a photo of shoes or sneakers",
        "a photo of clothing, t-shirt, or apparel",
        "a photo of electronics, headphones, or smartwatch",
        "a photo of accessories, backpack, or bag"
    ]
    inputs = clip_processor(
        text=category_prompts, 
        images=image, 
        return_tensors="pt", 
        padding=True
    )
    with torch.no_grad():
        outputs = clip_model(**inputs)
        
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1)
    best_category_idx = torch.argmax(probs, dim=1).item()
    categories = ["shoes", "apparel", "electronics", "accessories"]
    predicted_category = categories[best_category_idx]
    confidence = probs[0][best_category_idx].item()
    if confidence > 0.3:
        return predicted_category
    return None

def call_zephyr_llm(query: str) -> str:
    prompt = f"""You are Palona, a helpful AI assistant for an online shopping platform (like Amazon Rufus).
Your job is to explain your role and guide users about how you assist with finding products and answering shopping-related queries.
Avoid giving general knowledge or emotional/philosophical responses. Do not recommend products — that is handled elsewhere.
Respond only as Palona to the following message:

User: {query}
Palona:"""

    try:
        response = requests.post(
            "https://api-inference.huggingface.co/models/HuggingFaceH4/zephyr-7b-beta",
            headers={"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"},
            json={"inputs": prompt},
            timeout=30
        )
        if not response.content:
            print("Empty response from Hugging Face API.")
            return "Sorry, I'm having trouble responding right now. Please try again soon."
        try:
            result = response.json()
        except Exception:
            return "Sorry, I didn't understand the reply I received. Please try again."

        if isinstance(result, list) and "generated_text" in result[0]:
            full_text = result[0]["generated_text"]
            if "Palona:" in full_text:
                return full_text.split("Palona:")[-1].strip()
            else:
                return full_text.strip()
    except requests.exceptions.RequestException as e:
        print("Zephyr call failed:", e)
        return "Sorry, I couldn't connect to my assistant right now. Please try again later."


@app.post("/api/chat",tags=["AI Agent"], summary="Chat with Palona AI")
def chat(req: ChatRequest):
    query = req.message
    category = extract_category_from_query(query)
    if category:
        query_embedding = text_model.encode([query])
        relevant_products = [p for p in products if p["category"] == category]
        for p in relevant_products:
            p["score"] = cosine_similarity([query_embedding[0]], [p["text_embedding"]])[0][0]
        top_matches = sorted(relevant_products, key=lambda x: x["score"], reverse=True)[:3]
        return {
            "reply": f"Here are some {category} suggestions based on your query.",
            "products": [
                {
                    "title": p["title"],
                    "price": p["price"],
                    "image": p["image"],
                    "description": p["description"]
                }
                for p in top_matches
            ]
        }
    reply = call_zephyr_llm(query)
    return {
        "reply": reply,
        "products": []
    }

@app.post("/api/image-search",tags=["AI Agent"], summary="Search products using an image")
async def image_search(
    file: UploadFile = File(...),
    message: Optional[str] = Form(None)
):
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        predicted_category = predict_category_from_image(image)
        text_category = None
        if message:
            text_category = extract_category_from_query(message)
        category = text_category or predicted_category
        inputs = clip_processor(images=image, return_tensors="pt")
        with torch.no_grad():
            image_features = clip_model.get_image_features(**inputs)
            image_embedding = image_features.squeeze().numpy()
        if category:
            candidates = [p for p in products if p["category"] == category]
            category_message = f"I found products in the {category} category that match your image."
        else:
            candidates = products
            category_message = "Here are the most visually similar products I found."
        if not candidates:
            return {
                "reply": f"I couldn't find any products similar to your image.",
                "products": []
            }
        for p in candidates:
            p["score"] = cosine_similarity([image_embedding], [p["clip_embedding"]])[0][0]
        scored_candidates = [p for p in candidates if p["score"] > 0.2]
        top_matches = sorted(scored_candidates, key=lambda x: x["score"], reverse=True)[:3]
        if len(top_matches) < 2 and category:
            for p in products:
                p["score"] = cosine_similarity([image_embedding], [p["clip_embedding"]])[0][0]
            
            top_matches = sorted(products, key=lambda x: x["score"], reverse=True)[:3]
            category_message = "Here are products that look similar to your image."
        
        return {
            "reply": category_message,
            "products": [
                {
                    "title": p["title"],
                    "price": p["price"],
                    "image": p["image"],
                    "description": p["description"],
                }
                for p in top_matches
            ]
        }
    except Exception as e:
        print(f"Error in image search: {e}")
        return {
            "reply": "Sorry, I had trouble processing your image. Please try again.",
            "products": []
        }

@app.get("/")
def root():
    return FileResponse("frontend/chatui/build/index.html")