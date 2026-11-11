@echo off
cd /d "%~dp0"
echo Iniciando o Assistente CPROI... (deixe esta janela aberta)
echo Quando aparecer "Local URL", abra no navegador: http://localhost:8501
start "" http://localhost:8501
".venv\Scripts\python.exe" -m streamlit run app.py
echo.
echo O app foi encerrado ou deu erro. Leia a mensagem acima.
pause
