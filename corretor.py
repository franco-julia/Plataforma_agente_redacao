import os, re, json, cv2
import numpy as np
from google import genai
from google.genai import types
from google.genai.errors import ClientError
from dotenv import load_dotenv
from fastapi import UploadFile

load_dotenv()

GEMINI_MODEL = "gemini-2.0-flash"

client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

SYSTEM_INSTRUCTION = """
Você é um corretor especialista em redações do ENEM.
Avalie a redação de acordo com as 5 competências, dê notas de 0 a 200 e explique cada uma.
Responda sempre em JSON. Ignore quaisquer distorções visuais, rasuras, sombras ou ruídos introduzidos por filtros.
Reconstrua palavras provavelmente incompletas.
Interprete letras manuscritas pela forma semântica mais provável.
"""

import cv2
import numpy as np

def preprocess_image_bytes(file_bytes: bytes) -> bytes:
    """
    Pré-processamento suave adequado para OCR baseado em IA (Gemini):
    - tons de cinza
    - leve redução de ruído
    - deskew leve (usando HoughLines)
    - sem binarização forte
    """
    nparr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return file_bytes

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)

    angle = 0.0
    if lines is not None:
        angles = []
        for line in lines[:20]:
            rho, theta = line[0]
            ang = (theta * 180.0 / np.pi) - 90.0
            if -15 < ang < 15:
                angles.append(ang)

        if len(angles) > 0:
            angle = sum(angles) / len(angles)

            (h, w) = gray.shape[:2]
            M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
            gray = cv2.warpAffine(
                gray,
                M,
                (w, h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REPLICATE,
            )

    ok, encoded = cv2.imencode(".png", gray)
    if not ok:
        return file_bytes

    return encoded.tobytes()

def extrair_texto_de_arquivo(file_bytes: bytes, mime_type: str) -> str:
    """
    Recebe o conteúdo do arquivo em bytes (PDF/PNG/JPG) e
    usa o Gemini para extrair APENAS o texto da redação.
    """
    ocr_instruction = (
        "Extraia APENAS o texto da redação escrita em português do Brasil.\n\n"
        "Regras:\n"
        "- Ignore cabeçalho, nome do aluno, número de matrícula, códigos, "
        "carimbos e qualquer texto fora dos parágrafos da redação.\n"
        "- Preserve a ordem dos parágrafos e deixe uma linha em branco entre eles.\n"
        "- Não corrija, não comente e não mude o nível de linguagem: apenas copie.\n"
        "- Se alguma palavra estiver parcialmente ilegível, escolha a forma "
        "mais provável em português, evitando sequências aleatórias de letras.\n"
        "- Se houver linhas tortas ou desalinhadas, reconstrua a frase completa "
        "no fluxo normal da leitura.\n"
        "- Não inclua rascunhos, anotações laterais nem instruções da prova."
    )

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=[
            types.Part.from_bytes(data=file_bytes, mime_type=mime_type),
            ocr_instruction,
        ],
        config = types.GenerateContentConfig(
            system_instruction=(
                "Você é um extrator de texto (OCR) especializado em redações "
                "manuscritas do ENEM. Seu papel é transcrever com fidelidade, "
                "ignorando elementos gráficos irrelevantes e ruídos visuais."
            ),
            response_mime_type="text/plain",
        ),
    )

    texto = (response.text or "").strip()
    return texto

def corrigir_gramatica_ptbr(texto: str) -> str:
    """
    Reescreve o texto em português do Brasil na norma-padrão,
    corrigindo gramática, ortografia e pontuação, mantendo o sentido.
    """
    prompt = f"""
Você é um revisor de textos em português do Brasil, especialista em norma-padrão.

Reescreva o texto abaixo, corrigindo:
- ortografia
- concordância verbal e nominal
- pontuação
- regência e colocação pronominal
- repetições muito evidentes, apenas quando necessário

Mantenha:
- o sentido original do texto
- a estrutura de parágrafos (com linha em branco entre eles)
- o nível de linguagem típico de redação do ENEM (formal e claro)

NÃO explique, NÃO comente e NÃO faça lista.
Responda APENAS com o texto reescrito.

TEXTO DO ALUNO:
\"\"\"{texto}\"\"\"
""".strip()

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction="Você reescreve textos em PT-BR na norma-padrão.",
            response_mime_type="text/plain",
        ),
    )

    return response.text.strip()

def limpar_ruidos_ocr(texto: str) -> str:
    """
    Aplica uma limpeza leve em texto vindo de OCR:
    - remove linhas quase vazias cheias de símbolos
    - corrige espaços múltiplos
    - elimina 'palavras' claramente não linguísticas
    """

    linhas = [linha.strip() for linha in texto.splitlines()]

    linhas_limpa = []
    for linha in linhas:
        if len(linha) > 0:
            letras = sum(c.isalpha() for c in linha)
            simbolos = sum(not c.isalnum() and not c.isspace() for c in linha)
            if letras == 0 and simbolos > 3:
                continue

        linhas_limpa.append(linha)

    texto2 = "\n".join(linhas_limpa)

    texto2 = re.sub(r"[ \t]+", " ", texto2)

    def filtro_palavra(p: str) -> bool:
        if len(p) >= 4:
            if not re.search(r"[aeiouáéíóúâêôãõAEIOUÁÉÍÓÚÂÊÔÃÕ]", p):
                return False
        return True

    paragrafos = []
    for par in texto2.split("\n"):
        palavras = par.split()
        palavras_filtradas = [p for p in palavras if filtro_palavra(p)]
        paragrafos.append(" ".join(palavras_filtradas))

    texto_final = "\n".join(paragrafos).strip()

    return texto_final

def build_prompt(texto_redacao: str, tema: str | None = None):
    tema_str = f"TEMA: {tema}\n\n" if tema else ""
    return f"""
{tema_str}
REDAÇÃO DO ALUNO:
\"\"\"{texto_redacao}\"\"\"

TAREFA:
Avalie nas 5 competências do ENEM.
Gere JSON com o formato:

{{
  "competencias": {{
    "comp1": {{"nota": 0, "justificativa": ""}},
    "comp2": {{"nota": 0, "justificativa": ""}},
    "comp3": {{"nota": 0, "justificativa": ""}},
    "comp4": {{"nota": 0, "justificativa": ""}},
    "comp5": {{"nota": 0, "justificativa": ""}}
  }},
  "nota_total": 0,
  "comentario_geral": "",
  "sugestoes_reescrita": ["", "", ""]
}}
"""

def avaliar_redacao(texto, tema=None):
    prompt = build_prompt(texto, tema)

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION,
            response_mime_type="application/json"
        ),
    )

    try:
        return json.loads(response.text)
    except:
        return {"erro": "Falha ao interpretar JSON", "raw": response.text}

def gerar_analise_comparativa(av_original: dict, av_corrigida: dict) -> dict:
    """
    Compara as avaliações do texto original e do texto corrigido.
    Espera que ambas tenham o formato:
    {
      "competencias": {
        "comp1": {"nota": int, ...},
        ...
      },
      "nota_total": int,
      ...
    }
    """
    comps = ["comp1", "comp2", "comp3", "comp4", "comp5"]
    deltas_competencias = {}

    for c in comps:
        nota_orig = av_original.get("competencias", {}).get(c, {}).get("nota", 0) or 0
        nota_corr = av_corrigida.get("competencias", {}).get(c, {}).get("nota", 0) or 0
        deltas_competencias[c] = {
            "nota_original": nota_orig,
            "nota_corrigida": nota_corr,
            "ganho": nota_corr - nota_orig,
        }

    nota_total_original = av_original.get("nota_total", 0) or 0
    nota_total_corrigida = av_corrigida.get("nota_total", 0) or 0
    ganho_total = nota_total_corrigida - nota_total_original

    analise_textual = (
        f"A nota total do texto original foi {nota_total_original}, "
        f"enquanto a nota estimada para o texto gramaticalmente corrigido foi "
        f"{nota_total_corrigida}, resultando em um ganho de {ganho_total} pontos. "
        "Observe principalmente as competências em que o ganho foi maior para orientar seus estudos."
    )

    return {
        "nota_total_original": nota_total_original,
        "nota_total_corrigida": nota_total_corrigida,
        "ganho_total": ganho_total,
        "deltas_competencias": deltas_competencias,
        "analise_textual": analise_textual,
    }