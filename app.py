from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel

from corretor import (
    avaliar_redacao,
    corrigir_gramatica_ptbr,
    gerar_analise_comparativa,
)
from ocr import ler_redacao

app = FastAPI()

class RedacaoInput(BaseModel):
    texto: str
    tema: str | None = None

@app.post("/corrigir")
def corrigir_redacao(payload: RedacaoInput):

    avaliacao_original = avaliar_redacao(payload.texto, payload.tema)
    texto_corrigido = corrigir_gramatica_ptbr(payload.texto)
    avaliacao_corrigida = avaliar_redacao(texto_corrigido, payload.tema)
    comparativo = gerar_analise_comparativa(avaliacao_original, avaliacao_corrigida)

    return {
        "texto_original": payload.texto,
        "texto_corrigido": texto_corrigido,
        "tema": payload.tema,
        "avaliacao_original": avaliacao_original,
        "avaliacao_corrigida": avaliacao_corrigida,
        "comparativo": comparativo,
    }

@app.post("/corrigir-arquivo")
async def corrigir_redacao_arquivo(
    arquivo: UploadFile = File(...),
    tema: str | None = None,
):
    texto = await ler_redacao(arquivo)

    if not texto or len(texto.strip()) < 20:
        raise HTTPException(
            status_code=400,
            detail="Não foi possível extrair texto da redação. Envie uma imagem/PDF mais nítido.",
        )

    avaliacao_original = avaliar_redacao(texto, tema)
    texto_corrigido = corrigir_gramatica_ptbr(texto)
    avaliacao_corrigida = avaliar_redacao(texto_corrigido, tema)
    comparativo = gerar_analise_comparativa(avaliacao_original, avaliacao_corrigida)

    return {
        "texto_extraido": texto,
        "texto_corrigido": texto_corrigido,
        "tema": tema,
        "avaliacao_original": avaliacao_original,
        "avaliacao_corrigida": avaliacao_corrigida,
        "comparativo": comparativo,
    }
