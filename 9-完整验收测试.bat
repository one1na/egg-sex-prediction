@echo off
call "%~dp0code\run.bat" -m unittest discover -s acceptance -v
