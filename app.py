from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
import cv2
import numpy as np
from io import BytesIO
import base64
import json
import logging
from starlette.requests import Request

app = FastAPI()

# Setup Jinja2 Templates
templates = Jinja2Templates(directory="templates")

logging.basicConfig(level=logging.INFO)

@app.get("/", response_class=HTMLResponse)
async def main_page(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload")
async def upload_image(request: Request, file: UploadFile = File(...)):
    if not file:
        raise HTTPException(status_code=400, detail="No file uploaded")

    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image data")
        
        _, buffer = cv2.imencode('.jpg', img)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
    except Exception as e:
        logging.error(f"Error processing image: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

    return templates.TemplateResponse("crop.html", {"request": request, "img_base64": img_base64})

@app.post("/crop")
async def crop_image(image: str = Form(...), cropData: str = Form(...)):
    if not image or not cropData:
        raise HTTPException(status_code=400, detail="Missing image or crop data")

    try:
        # Decode the base64 image data
        img_data = base64.b64decode(image.split(",")[1])
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image data")

        # Parse the cropData JSON
        crop_data = json.loads(cropData)
        x = int(crop_data['x'])
        y = int(crop_data['y'])
        width = int(crop_data['width'])
        height = int(crop_data['height'])
        
        # Crop the image
        cropped_img = img[y:y+height, x:x+width]

        # Save cropped image as JPEG
        _, buffer = cv2.imencode('.jpg', cropped_img)
        if buffer is None:
            raise HTTPException(status_code=500, detail="Error encoding cropped image")

        # Convert to bytes
        img_bytes = buffer.tobytes()

        # Create a streaming response for the download
        return StreamingResponse(BytesIO(img_bytes), media_type="image/jpeg", headers={"Content-Disposition": "attachment; filename=cropped_image.jpg"})
    except Exception as e:
        logging.error(f"Error cropping image: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error cropping image: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
