@echo off
REM Starts the Assignment 3 explorer server.
REM Assumes your virtual environment is already created and activated,
REM and that `pip install -r requirements.txt` has been run in it.
cd /d "%~dp0"
python run_sandbox.py
