@echo off
echo Installing dependencies...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo.
    echo pip failed. Trying pip3...
    pip3 install -r requirements.txt
)
echo.
echo Starting app...
streamlit run streamlit_app.py
