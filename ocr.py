import io
from typing import Any
import numpy as np
import cv2
import fitz  # type: ignore # PyMuPDF
from PIL import Image
import pytesseract # type: ignore
from google.genai import types
from google.genai.errors import ClientError
from corretor import client, GEMINI_MODEL

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

OCR_PROMPT = """
Você está vendo a imagem de uma redação manuscrita do ENEM.

Regras:
- Ignore completamente cabeçalho, logos, margens, sombras e teclado.
- Leia SOMENTE as linhas manuscritas do texto.
- As linhas do caderno (linhas vermelhas ou cinzas) NÃO são texto.
- Preserve os parágrafos exatamente como aparecem (linha em branco entre eles).
- Se alguma palavra estiver pouco legível, tente inferir sem inventar frases inteiras.
- Não adicione comentários, títulos ou observações.
- Retorne APENAS o texto puro da redação em português.
""".strip()

def preprocess_image_for_ocr(file_bytes: bytes) -> bytes:
    """
    Pré-processa a imagem para melhorar o OCR:
    - crop da parte superior (onde costuma ter sombra/cabeçalho)
    - conversão para cinza
    - deskew (correção de inclinação)
    - equalização de histograma (contraste)
    - binarização adaptativa
    - remoção de linhas da folha
    - dilatação leve da escrita
    """
    nparr = np.frombuffer(file_bytes, np.uint8)
    img_color = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img_color is None:
        return file_bytes

    h, w, _ = img_color.shape

    crop_y = int(h * 0.25)
    img_color = img_color[crop_y:h, 0:w]
    gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

    coords = np.column_stack(np.where(gray > 0))
    if coords.size > 0:
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
        h2, w2 = gray.shape
        M = cv2.getRotationMatrix2D((w2 // 2, h2 // 2), angle, 1.0)
        gray = cv2.warpAffine(gray, M, (w2, h2), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    gray = cv2.equalizeHist(gray)

    bw = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        8,
    )

    kernel = np.ones((1, 50), np.uint8)  # kernel horizontal
    lines = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel)
    bw_no_lines = cv2.subtract(bw, lines)

    kernel2 = np.ones((2, 2), np.uint8)
    final = cv2.dilate(bw_no_lines, kernel2, iterations=1)

    success, buf = cv2.imencode(".png", final)
    if not success:
        return file_bytes
    return buf.tobytes()

#  PDF: DETECTAR TEXTO NATIVO x ESCANEADO
def pdf_possui_texto(pdf_bytes: bytes) -> bool:
    """
    Verifica se o PDF possui texto embutido (PDF nativo).
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    for page in doc:
        if page.get_text().strip():
            return True
    return False

def render_pdf_to_image(pdf_bytes: bytes) -> bytes:
    """
    Renderiza a primeira página do PDF como imagem PNG em alta resolução.
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page = doc.load_page(0)
    pix = page.get_pixmap(dpi=300)
    return pix.tobytes("png")

# OCR COM GEMINI
def ocr_gemini(file_bytes: bytes, mime_type: str) -> str:
    """
    Tenta extrair texto usando Gemini com prompt orientado a redação.
    """
    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=[
                types.Part.from_bytes(data=file_bytes, mime_type=mime_type),
                OCR_PROMPT,
            ],
            config=types.GenerateContentConfig(
                system_instruction="Você é um OCR avançado especializado em redações manuscritas.",
                response_mime_type="text/plain",
            ),
        )
    except ClientError:
        return ""

    texto = (response.text or "").strip()
    return texto

# OCR 
def ocr_tesseract(file_bytes: bytes) -> str:
    """
    Fallback de OCR usando Tesseract configurado para manuscrito em português.
    """
    try:
        img = Image.open(io.BytesIO(file_bytes))
    except Exception:
        return ""

    custom_config = r"--oem 1 --psm 6 -c preserve_interword_spaces=1"
    try:
        texto = pytesseract.image_to_string(img, lang="por", config=custom_config)
    except Exception:
        return ""

    return (texto or "").strip()

async def ler_redacao(arquivo: Any) -> str:
    """
    Pipeline completo de leitura:

    1. Lê bytes do UploadFile.
    2. Se PDF:
       - Se tiver texto nativo → extrai direto (sem Gemini).
       - Se não tiver texto → renderiza como imagem e segue fluxo de imagem.
    3. Se imagem (JPG/PNG):
       - Pré-processa.
       - Tenta Gemini OCR.
       - Se falhar ou vier pouco texto → Tesseract como fallback.
    """
    file_bytes = await arquivo.read()
    if not file_bytes:
        return ""

    content_type = getattr(arquivo, "content_type", None) or "application/octet-stream"

    if content_type == "application/pdf":
        if pdf_possui_texto(file_bytes):
            doc = fitz.open(stream=file_bytes, filetype="pdf")
            texto_pdf = []
            for page in doc:
                texto_pdf.append(page.get_text())
            texto = "\n".join(texto_pdf).strip()
            return texto

        file_bytes = render_pdf_to_image(file_bytes)
        content_type = "image/png"

    if content_type in ("image/jpeg", "image/png"):
        processed = preprocess_image_for_ocr(file_bytes)

        texto = ocr_gemini(processed, "image/png")
        if texto and len(texto.strip()) > 50:
            return texto

        texto_fallback = ocr_tesseract(processed)
        if texto_fallback:
            return texto_fallback

        return ""

    return ""
