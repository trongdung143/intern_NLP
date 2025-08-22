import pdfplumber
from pathlib import Path
from tqdm import tqdm


def clean_text(text: str) -> str:
    text = text.replace("|", " ")
    text = text.replace("\n", " ").replace("\r", "")
    text = "".join(ch for ch in text if ch.isprintable())
    return text


def convert_pdf_folder_to_txt(pdf_folder, txt_output_folder):
    pdf_folder = Path(pdf_folder)
    txt_output_folder = Path(txt_output_folder)
    txt_output_folder.mkdir(parents=True, exist_ok=True)

    for pdf_file in tqdm(pdf_folder.glob("*.pdf")):
        with pdfplumber.open(pdf_file) as pdf:
            text = "\n".join(page.extract_text() or "" for page in pdf.pages)

        cleaned = clean_text(text)
        txt_file = txt_output_folder / (pdf_file.stem + ".txt")
        txt_file.write_text(cleaned.strip(), encoding="utf-8")
        print(f"[✓] Đã tạo: {txt_file.name}")


def convert_single_pdf_to_txt(pdf_path, txt_output_path=None):
    pdf_path = Path(pdf_path)
    with pdfplumber.open(pdf_path) as pdf:
        text = "\n".join(page.extract_text() or "" for page in pdf.pages)
    cleaned = clean_text(text)

    if txt_output_path:
        txt_file = Path(txt_output_path)
        txt_file.write_text(cleaned.strip(), encoding="utf-8")
        print(f"[✓] Đã tạo: {txt_file}")
    return cleaned


convert_pdf_folder_to_txt(
    "./data/resumes_additional/", "./data/resumes_additional_txt/"
)
