from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from predict import predict
import pdfplumber
import io
import os

app = FastAPI(title="CV NER API", description="API để trích xuất thông tin từ CV")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.mount("/static", StaticFiles(directory="static"), name="static")


def clean_text(text: str) -> str:
    text = text.replace("|", " ")
    text = text.replace("\n", " ").replace("\r", "")
    text = "".join(ch for ch in text if ch.isprintable())
    return text


def convert_single_pdf_to_txt_bytes(pdf_bytes: bytes) -> str:
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        text = "\n".join(page.extract_text() or "" for page in pdf.pages)
    return clean_text(text)


@app.post("/extract-cv")
async def extract_cv_info(file: UploadFile = File(...)):
    try:
        content = await file.read()
        text = convert_single_pdf_to_txt_bytes(content)

        if not text.strip():
            raise HTTPException(
                status_code=400, detail="Không trích xuất được nội dung từ file."
            )

        result = predict(text)

        return JSONResponse(
            content={
                "success": True,
                "email": result["email"],
                "name": result["name"],
                "message": "Xử lý thành công",
            }
        )

    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi server: {str(e)}")


@app.get("/health")
async def health_check():
    return {"status": "healthy", "message": "CV NER API đang hoạt động"}


@app.get("/", response_class=HTMLResponse)
async def root():
    try:
        with open("static/index.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read(), status_code=200)
    except FileNotFoundError:
        return HTMLResponse(content="Không tìm thấy index.html", status_code=404)
