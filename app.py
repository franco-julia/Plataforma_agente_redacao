from fastapi import FastAPI
from pydantic import BaseModel
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from corretor import (
    avaliar_redacao,
    corrigir_gramatica_ptbr,
    extrair_texto_de_arquivo,
    gerar_analise_comparativa,
    preprocess_image_bytes,
    limpar_ruidos_ocr,
)

app = FastAPI()

app.mount("/static", StaticFiles(directory="frontend"), name="static")

@app.get("/", response_class=HTMLResponse)
def read_root():
    with open("frontend/index.html", "r", encoding="utf-8") as f:
        return f.read()

class RedacaoInput(BaseModel):
    texto: str
    tema: str | None = None

class RedacaoInput(BaseModel):
    texto: str
    tema: str | None = None

@app.post("/corrigir-arquivo")
async def corrigir_redacao_arquivo(
    arquivo: UploadFile = File(...),
    tema: str | None = None
):
    if arquivo.content_type not in [
        "application/pdf",
        "image/png",
        "image/jpeg",
    ]:
        raise HTTPException(
            status_code=400,
            detail=f"Tipo de arquivo não suportado: {arquivo.content_type}",
        )

    file_bytes = await arquivo.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Arquivo vazio.")

    mime_type_para_ocr = arquivo.content_type
    if arquivo.content_type in ["image/png", "image/jpeg"]:
        file_bytes = preprocess_image_bytes(file_bytes)
        mime_type_para_ocr = "image/png"

    texto = extrair_texto_de_arquivo(file_bytes, mime_type_para_ocr)
    texto_limpo = limpar_ruidos_ocr(texto)

    if not texto or len(texto.strip()) == 0:
        raise HTTPException(
            status_code=400,
            detail="Não foi possível extrair texto da redação (OCR retornou vazio).",
        )

    avaliacao_original = avaliar_redacao(texto_limpo, tema)
    texto_corrigido = corrigir_gramatica_ptbr(texto_limpo)
    avaliacao_corrigida = avaliar_redacao(texto_corrigido, tema)
    comparativo = gerar_analise_comparativa(avaliacao_original, avaliacao_corrigida)

    return {
        "texto_extraido": texto,
        "texto_corrigido": texto_corrigido,
        "tema": tema,
        "avaliacao_original": avaliacao_original,
        "avaliacao_texto_corrigido": avaliacao_corrigida,
        "comparativo": comparativo,
    }

@app.post("/corrigir")
def corrigir_redacao(payload: RedacaoInput):
    texto_limpo = limpar_ruidos_ocr(payload.texto)

    avaliacao_original = avaliar_redacao(texto_limpo, payload.tema)
    texto_corrigido = corrigir_gramatica_ptbr(texto_limpo)
    avaliacao_corrigida = avaliar_redacao(texto_corrigido, payload.tema)

    comparativo = gerar_analise_comparativa(avaliacao_original, avaliacao_corrigida)

    return {
        "texto_original": payload.texto,
        "texto_pos_ocr_limpo": texto_limpo,
        "texto_corrigido": texto_corrigido,
        "tema": payload.tema,
        "avaliacao_original": avaliacao_original,
        "avaliacao_texto_corrigido": avaliacao_corrigida,
        "comparativo": comparativo,
    }
