@echo off
cd /d "%~dp0"
echo Installing packages...
python -m pip install --quiet numpy scipy pandas scikit-learn matplotlib qiskit qiskit-aer
echo.
echo Running Q4...
python q4_bus_maintenance.py > q4_output.txt 2>&1
type q4_output.txt
echo.
echo Output also saved to q4_output.txt
pause
