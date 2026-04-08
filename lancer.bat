@echo off
echo ================================================
echo    Lancement AI Early Disease Prediction
echo ================================================
echo.

echo [1] Lancement du Backend Flask...
start "Backend Flask" cmd /k "python api.py"

echo [2] Lancement du Frontend Streamlit...
start "Frontend Streamlit" cmd /k "python -m streamlit run app.py"

echo.
echo Les deux fenêtres sont lancées...
echo Attends que Streamlit ouvre le navigateur.
pause