# 📚 Assistente de Estudo com IA Local (RAG)

Assistente que lê um PDF de estudo e responde perguntas **usando apenas o conteúdo do material**, citando as páginas de origem. Também tem um **modo quiz**: gera perguntas a partir de trechos do PDF e corrige as suas respostas.

Roda **100% local** com [Ollama](https://ollama.com), sem enviar dados para a nuvem e sem custo de API.

## Funcionalidades

- 💬 **Perguntar ao PDF**: respostas em streaming, com as fontes (páginas e trechos) em cada resposta.
- 📝 **Modo quiz**: pergunta gerada a partir de um trecho aleatório, com correção (veredito, comentário e resposta esperada).
- ⚡ **Índice persistente**: o PDF é indexado uma vez; se o arquivo mudar, o índice é refeito automaticamente.
- 🔄 **Modelo trocável**: escolha qualquer modelo instalado no Ollama pela barra lateral.

## Como funciona

```
PDF → divisão em blocos → embeddings (Ollama) → índice FAISS
                                                     │
Pergunta → busca por similaridade → trechos relevantes → LLM (Ollama) → resposta + fontes
```

## Tecnologias

Python · Streamlit · LangChain · FAISS · Ollama · PyPDF

## Como rodar

1. Instale o [Ollama](https://ollama.com) e baixe os modelos:
   ```bash
   ollama pull qwen3-embedding:0.6b
   ollama pull qwen3.5:4b
   ```
   Se ficar lento, use um modelo menor, como `qwen3.5:0.8b`, pela barra lateral do app.

2. Clone o repositório e instale as dependências:
   ```bash
   git clone https://github.com/JoaoJordao27/Assistente_CPROI.git
   cd Assistente_CPROI
   python -m venv .venv
   .venv\Scripts\activate        # Windows
   pip install -r requirements.txt
   ```

3. Coloque o **seu PDF** na pasta do projeto (o material de estudo não está incluído no repositório).

4. Inicie o app:
   ```bash
   streamlit run app.py
   ```

Na primeira execução o índice do PDF é criado na pasta `indice/`; nas seguintes o app abre direto.

## Licença

[MIT](LICENSE)
