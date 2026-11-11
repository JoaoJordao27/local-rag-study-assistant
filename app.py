"""
Assistente de estudo para a prova (CPROI) - 100% local.

- Lê o PDF que estiver na pasta do projeto
- Responde perguntas usando APENAS o conteúdo do PDF (RAG)
- Modo quiz: gera perguntas a partir do PDF e corrige suas respostas

Tecnologias: Streamlit + LangChain + FAISS + Ollama (sem internet).
Para rodar:  streamlit run app.py
"""

import json
import random
from pathlib import Path

import ollama
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

PASTA = Path(__file__).parent
PASTA_INDICE = PASTA / "indice"
MODELO_EMBEDDING = "qwen3-embedding:0.6b"
MODELO_LLM_PADRAO = "qwen3.5:4b"
TAMANHO_BLOCO = 1000
SOBREPOSICAO = 200
BLOCOS_NA_BUSCA = 6


# ----------------------------------------------------------------------------
# PDF e índice vetorial
# ----------------------------------------------------------------------------
def listar_pdfs():
    return sorted(PASTA.glob("*.pdf"))


def assinatura_pdf(pdf: Path):
    """Identifica a versão do PDF: se mudar, o índice é refeito."""
    info = pdf.stat()
    return {
        "arquivo": pdf.name,
        "tamanho": info.st_size,
        "modificado": int(info.st_mtime),
        "embedding": MODELO_EMBEDDING,
        "bloco": TAMANHO_BLOCO,
        "sobreposicao": SOBREPOSICAO,
    }


def criar_embeddings():
    return OllamaEmbeddings(model=MODELO_EMBEDDING)


def ler_blocos(pdf: Path):
    paginas = PyPDFLoader(str(pdf)).load()
    separador = RecursiveCharacterTextSplitter(
        chunk_size=TAMANHO_BLOCO, chunk_overlap=SOBREPOSICAO
    )
    blocos = separador.split_documents(paginas)
    # descarta blocos quase vazios (cabeçalhos, números de página etc.)
    return [b for b in blocos if len(b.page_content.strip()) > 80]


@st.cache_resource(show_spinner=False)
def carregar_indice(caminho_pdf: str):
    """Carrega o índice salvo em disco ou cria um novo se o PDF mudou."""
    pdf = Path(caminho_pdf)
    assinatura = assinatura_pdf(pdf)
    arquivo_meta = PASTA_INDICE / "meta.json"
    embeddings = criar_embeddings()

    if arquivo_meta.exists():
        try:
            if json.loads(arquivo_meta.read_text(encoding="utf-8")) == assinatura:
                # o índice foi criado por este próprio app, então é seguro carregar
                # (bytes via Python: o FAISS em C++ falha com acentos no caminho)
                return FAISS.deserialize_from_bytes(
                    (PASTA_INDICE / "indice.bin").read_bytes(),
                    embeddings,
                    allow_dangerous_deserialization=True,
                )
        except Exception:
            pass  # índice corrompido: refaz abaixo

    blocos = ler_blocos(pdf)
    if not blocos:
        raise ValueError(
            "Não consegui extrair texto desse PDF. Ele pode ser escaneado (imagem)."
        )

    barra = st.progress(0.0, text="Criando o índice do PDF (só na primeira vez)...")
    lote = 16
    indice = None
    for i in range(0, len(blocos), lote):
        parte = blocos[i : i + lote]
        if indice is None:
            indice = FAISS.from_documents(parte, embeddings)
        else:
            indice.add_documents(parte)
        barra.progress(min((i + lote) / len(blocos), 1.0))
    barra.empty()

    PASTA_INDICE.mkdir(exist_ok=True)
    (PASTA_INDICE / "indice.bin").write_bytes(indice.serialize_to_bytes())
    arquivo_meta.write_text(json.dumps(assinatura), encoding="utf-8")
    return indice


def todos_os_blocos(indice):
    return list(indice.docstore._dict.values())


def pagina(doc):
    return int(doc.metadata.get("page", 0)) + 1


# ----------------------------------------------------------------------------
# Modelo de linguagem e prompts
# ----------------------------------------------------------------------------
def modelos_instalados():
    """Modelos de chat baixados no Ollama (ignora os de embedding), do menor ao maior."""
    try:
        lista = ollama.list().models
    except Exception:
        return []
    chat = [m for m in lista if "embed" not in m.model.lower()]
    return [m.model for m in sorted(chat, key=lambda m: m.size or 0)]


def criar_llm(modelo: str, temperatura: float):
    return ChatOllama(model=modelo, reasoning=False, temperature=temperatura)


PROMPT_RESPOSTA = ChatPromptTemplate.from_template(
    """Você é um assistente de estudos que responde perguntas sobre o material de uma disciplina.
Use APENAS as informações do contexto abaixo para responder.
Se a resposta não estiver no contexto, diga claramente que não encontrou no material.
Responda em português do Brasil, de forma clara e objetiva.

Contexto:
{contexto}

Pergunta: {pergunta}

Resposta:"""
)

PROMPT_GERAR_PERGUNTA = ChatPromptTemplate.from_template(
    """Você é um professor elaborando uma questão de prova.
Com base APENAS no trecho abaixo, escreva UMA pergunta discursiva curta, clara e que
possa ser respondida usando somente esse trecho.
Escreva apenas a pergunta, em português do Brasil, sem resposta e sem explicações.

Trecho:
{trecho}

Pergunta:"""
)

PROMPT_CORRIGIR = ChatPromptTemplate.from_template(
    """Você é um professor corrigindo a resposta de um aluno.
Use APENAS o trecho do material abaixo como referência de resposta correta.

Trecho do material:
{trecho}

Pergunta: {pergunta}

Resposta do aluno: {resposta}

Corrija seguindo exatamente este formato, em português do Brasil:
Veredito: (Correta, Parcialmente correta ou Incorreta)
Comentário: (explique em 1 ou 2 frases o que está certo ou faltou)
Resposta esperada: (a resposta correta, baseada no trecho)"""
)


def formatar_contexto(docs):
    return "\n\n".join(f"[Página {pagina(d)}]\n{d.page_content}" for d in docs)


def mostrar_fontes(docs):
    paginas = sorted({pagina(d) for d in docs})
    with st.expander(f"Fontes no PDF (páginas {', '.join(map(str, paginas))})"):
        for d in docs:
            st.markdown(f"**Página {pagina(d)}**")
            st.caption(d.page_content)


# ----------------------------------------------------------------------------
# Interface
# ----------------------------------------------------------------------------
st.set_page_config(page_title="Assistente CPROI", page_icon="📚")
st.title("📚 Assistente de estudo — CPROI")

pdfs = listar_pdfs()
if not pdfs:
    st.warning(
        f"Nenhum PDF encontrado. Coloque o PDF da matéria nesta pasta e recarregue a página:\n\n`{PASTA}`"
    )
    st.stop()

with st.sidebar:
    st.header("Configurações")
    if len(pdfs) > 1:
        nome_pdf = st.selectbox("PDF", [p.name for p in pdfs])
        pdf_escolhido = PASTA / nome_pdf
    else:
        pdf_escolhido = pdfs[0]
        st.write(f"PDF: **{pdf_escolhido.name}**")
    modelos = modelos_instalados()
    if modelos:
        modelo_llm = st.selectbox(
            "Modelo de IA",
            modelos,
            index=modelos.index(MODELO_LLM_PADRAO) if MODELO_LLM_PADRAO in modelos else 0,
            help="Menor = mais rápido; maior = respostas melhores, porém mais lento.",
        )
    else:
        st.warning("Não consegui listar os modelos. O Ollama está aberto?")
        modelo_llm = st.text_input("Modelo do Ollama", MODELO_LLM_PADRAO)
    st.caption("Para ter mais opções: `ollama pull <nome>` e recarregue a página.")

try:
    indice = carregar_indice(str(pdf_escolhido))
except Exception as erro:
    st.error(
        "Não foi possível preparar o PDF. Confira se o Ollama está aberto e se o modelo "
        f"`{MODELO_EMBEDDING}` está baixado.\n\nDetalhe: {erro}"
    )
    st.stop()

aba_perguntar, aba_quiz = st.tabs(["💬 Perguntar ao PDF", "📝 Modo quiz"])

# ---- Aba 1: perguntas e respostas -------------------------------------------
with aba_perguntar:
    if "mensagens" not in st.session_state:
        st.session_state["mensagens"] = []

    for msg in st.session_state["mensagens"]:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if msg.get("docs"):
                mostrar_fontes(msg["docs"])

    pergunta = st.chat_input("Pergunte algo sobre o conteúdo do PDF...")
    if pergunta:
        st.session_state["mensagens"].append({"role": "user", "content": pergunta})
        with st.chat_message("user"):
            st.write(pergunta)

        with st.chat_message("assistant"):
            try:
                docs = indice.similarity_search(pergunta, k=BLOCOS_NA_BUSCA)
                cadeia = PROMPT_RESPOSTA | criar_llm(modelo_llm, 0.1) | StrOutputParser()
                resposta = st.write_stream(
                    cadeia.stream(
                        {"contexto": formatar_contexto(docs), "pergunta": pergunta}
                    )
                )
                mostrar_fontes(docs)
                st.session_state["mensagens"].append(
                    {"role": "assistant", "content": resposta, "docs": docs}
                )
            except Exception as erro:
                st.error(f"Erro ao gerar a resposta. O Ollama está aberto?\n\n{erro}")

# ---- Aba 2: quiz --------------------------------------------------------------
with aba_quiz:
    st.write(
        "O assistente cria uma pergunta a partir de um trecho aleatório do PDF e "
        "corrige a sua resposta."
    )

    if st.button("🎲 Nova pergunta", type="primary"):
        blocos = [b for b in todos_os_blocos(indice) if len(b.page_content) > 300]
        trecho = random.choice(blocos or todos_os_blocos(indice))
        try:
            with st.spinner("Elaborando a pergunta..."):
                cadeia = PROMPT_GERAR_PERGUNTA | criar_llm(modelo_llm, 0.5) | StrOutputParser()
                texto = cadeia.invoke({"trecho": trecho.page_content}).strip()
            st.session_state["quiz"] = {"pergunta": texto, "trecho": trecho, "correcao": None}
            st.session_state["quiz_resposta"] = ""
        except Exception as erro:
            st.error(f"Erro ao gerar a pergunta. O Ollama está aberto?\n\n{erro}")

    quiz = st.session_state.get("quiz")
    if quiz:
        st.subheader("Pergunta")
        st.write(quiz["pergunta"])

        resposta_aluno = st.text_area("Sua resposta", key="quiz_resposta", height=150)

        if st.button("✅ Corrigir"):
            if not resposta_aluno.strip():
                st.warning("Escreva uma resposta antes de corrigir.")
            else:
                try:
                    with st.spinner("Corrigindo..."):
                        cadeia = PROMPT_CORRIGIR | criar_llm(modelo_llm, 0.1) | StrOutputParser()
                        quiz["correcao"] = cadeia.invoke(
                            {
                                "trecho": quiz["trecho"].page_content,
                                "pergunta": quiz["pergunta"],
                                "resposta": resposta_aluno,
                            }
                        )
                except Exception as erro:
                    st.error(f"Erro ao corrigir. O Ollama está aberto?\n\n{erro}")

        if quiz.get("correcao"):
            st.subheader("Correção")
            st.write(quiz["correcao"])
            mostrar_fontes([quiz["trecho"]])
