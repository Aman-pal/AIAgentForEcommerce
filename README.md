# Palona AI – Shopping Assistant

Palona AI is an intelligent and engaging shopping assistant that responds to inquiries about the store, lets users explore product details, and conducts visual searches using photographs. It imitates conversational AI, such as Amazon Rufus, and is intended for use in e-commerce.

---

## Project Structure

```
Palona_AI/
├── backend/
│   ├── main.py                # fastAPI backend with Zephyr and CLIP integration
│   ├── products.json          # contains metadata + descriptions for sample products
│   ├── requirements.txt      
│   ├── images/                # product image assets used for image-based matching
│   └── frontend_build/        # for potential hosting
│
└── frontend/
    └── chatui/               
        ├── public/            
        ├── src/               
        │   ├── App.js         # main component
        │   ├── index.js
        │   └── index.css      
        ├── .gitignore         
        ├── package.json       
        └── package-lock.json

```
## Frontend (React)

- **Framework**: React.js (bootstrapped using Create React App)
- **UI Library**: Material UI (MUI)
- **Functionality**:
  - conversational UI with message bubbles for assistant and user
  - file upload support (image search)
  - product cards rendered based on response
  - optimized layout using MUI’s responsive components
  - dynamic “typing” indicator

**Directory**: `frontend/chatui`

---

## Backend (FastAPI)

- **Framework**: FastAPI
- **LLM**: HuggingFace Zephyr-7B (used via Hugging Face Inference API)
- **Image Search**: OpenAI CLIP model for vision-text embedding
- **Text Matching**: sentenceTransformer for cosine similarity search
- **Endpoints**:
  - `POST /api/chat` → handles user text input and returns relevant products or a conversational reply
  - `POST /api/image-search` → accepts an image (and optionally a message), returns visually similar product suggestions

**File**: `backend/main.py`

---

## Product Data

`products.json` contains static product entries with fields like `title`, `description`, `price`, `image`, and `category`. During backend startup, it precomputes:

- SentenceTransformer embeddings for text search
- CLIP embeddings for image-text similarity

This approach allows fast matching against text or visual queries without a live database.

---

## Architecture Overview

- When a user types a product-related query, the backend uses:
  - keyword extraction to match a category
  - cosine similarity on precomputed embeddings to find top 3 matches
- When a user uploads an image:
  - CLIP predicts category and image embeddings
  - products are ranked based on visual similarity using cosine similarity
- For general queries (like “Who are you?”), the Zephyr LLM is invoked to generate a reply

---

## Decisions

- I used FastAPI for the backend because it's fast, easy to work with and great for building clean APIs.
- For understanding text messages from users, I used the Zephyr-7B model through Hugging Face—it gives good responses and doesn't require running heavy models locally.
- To handle image-based product search, I went with the CLIP model, which is really good at finding visual similarities between what the user uploads and what’s in the product list.
- For matching user queries with product info, I used SentenceTransformer—it gives quick and accurate results.
- On the frontend, I picked React along with Material UI to keep the interface simple, clean, and responsive. Everything was set up in a way that would let users chat naturally, upload product images, and get       smart suggestions easily

---

## Deployment Note

This project was initially attempted to be deployed using Render’s free tier. However, due to heavy model dependencies and memory usage (CLIP + SentenceTransformer + Hugging Face LLM), a higher compute plan is required. The deployment was paused due to these constraints.

---
