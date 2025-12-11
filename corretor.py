import os
import json
from typing import Dict, Any, Optional

from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.genai.errors import ClientError

load_dotenv()

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("Defina GEMINI_API_KEY no arquivo .env ou nas variáveis de ambiente.")

GEMINI_MODEL = "gemini-2.0-flash"
client = genai.Client(api_key=GEMINI_API_KEY)

#  ENEM
SYSTEM_INSTRUCTION_ENEM = """
Você é um corretor especialista em redações do ENEM.
Avalie a redação de acordo com as 5 competências do ENEM (0 a 200 cada):

Comp 1: domínio da norma padrão.
Comp 2: compreensão da proposta e organização das ideias.
Comp 3: seleção e organização de argumentos.
Comp 4: coesão e coerência na articulação do texto.
Comp 5: proposta de intervenção detalhada, respeitando direitos humanos.

Regras:
- Seja objetivo e técnico, mas em linguagem acessível ao estudante.
- Pode citar trechos da redação quando necessário.
- Sempre responda em JSON exatamente no formato solicitado.
"""

def build_prompt_avaliacao(essay_text: str, tema: Optional[str] = None) -> str:
    tema_str = f"TEMA: {tema}\n\n" if tema else ""
    return f"""
{tema_str}
REDAÇÃO DO ALUNO:
\"\"\"{essay_text}\"\"\"

TAREFA:
1. Atribua uma nota (0 a 200) para cada competência.
2. Justifique cada nota brevemente, apontando pontos fortes e pontos a melhorar.
3. Calcule a nota total (soma das 5 competências).
4. Dê um comentário geral.
5. Sugira 3 ações concretas para o aluno melhorar na próxima redação.

RESPONDA OBRIGATORIAMENTE EM JSON COM O SEGUINTE FORMATO:

{{
  "competencias": {{
    "comp1": {{"nota": 0, "justificativa": "..." }},
    "comp2": {{"nota": 0, "justificativa": "..." }},
    "comp3": {{"nota": 0, "justificativa": "..." }},
    "comp4": {{"nota": 0, "justificativa": "..." }},
    "comp5": {{"nota": 0, "justificativa": "..." }}
  }},
  "nota_total": 0,
  "comentario_geral": "...",
  "sugestoes_reescrita": ["...", "...", "..."]
}}
""".strip()

def avaliar_redacao(essay_text: str, tema: Optional[str] = None) -> Dict[str, Any]:
    """
    Avalia a redação em formato ENEM, retornando um JSON com notas e comentários.
    """
    prompt = build_prompt_avaliacao(essay_text, tema)

    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_INSTRUCTION_ENEM,
                response_mime_type="application/json",
            ),
        )
    except ClientError as e:
        raise e

    try:
        data = json.loads(response.text)
    except Exception:
        data = {
            "error": "Falha ao interpretar JSON de avaliação.",
            "raw_response": response.text,
        }

    return data

#  correção
SYSTEM_INSTRUCTION_GRAM = "Você reescreve textos em português do Brasil na norma-padrão."

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
- a estrutura de parágrafos (linha em branco entre eles)
- o nível de linguagem típico de redação do ENEM (formal e claro)

NÃO explique, NÃO comente e NÃO faça lista.
Responda APENAS com o texto reescrito.

TEXTO DO ALUNO:
\"\"\"{texto}\"\"\"
""".strip()

    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_INSTRUCTION_GRAM,
                response_mime_type="text/plain",
            ),
        )
    except ClientError as e:
        raise e

    return (response.text or "").strip()

#  comparação
def gerar_analise_comparativa(
    av_original: Dict[str, Any],
    av_corrigida: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Compara avaliações do texto original e do texto corrigido.
    Espera que ambas tenham o formato:
    {
      "competencias": { "comp1": {"nota": int, ...}, ... },
      "nota_total": int,
      ...
    }
    """
    comps = ["comp1", "comp2", "comp3", "comp4", "comp5"]
    deltas_competencias: Dict[str, Dict[str, Any]] = {}

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
