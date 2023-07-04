from fastapi import FastAPI, UploadFile, File
import os
from engine import load_image, get_embeddings, compare_embeddings

app = FastAPI()

@app.get("/")
def home():
    return "Face Recognition API"

@app.post("/compare-two-faces")
async def compare_two_faces(face1: UploadFile = File(...), face2: UploadFile = File(...)):
    try:
        # Save uploaded files to disk
        face1_path = "face1.jpg"
        face2_path = "face2.jpg"
        with open(face1_path, "wb") as f1, open(face2_path, "wb") as f2:
            f1.write(await face1.read())
            f2.write(await face2.read())

        # Load and preprocess images
        img1 = load_image(face1_path)
        img2 = load_image(face2_path)

        # Get embeddings
        emb1 = get_embeddings(img1)
        emb2 = get_embeddings(img2)

        # Compare embeddings
        result, similarity = compare_embeddings(emb1, emb2)

        # Remove temporary image files
        os.remove(face1_path)
        os.remove(face2_path)

        return {"result": result, "similarity": similarity}

    except Exception as e:
        return {"error": str(e)}
