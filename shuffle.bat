@ECHO OFF
SETLOCAL ENABLEDELAYEDEXPANSION
SET TmpFile=tmp%RANDOM%%RANDOM%.tmp
TYPE NUL >%Tmpfile%
FOR /F "tokens=*" %%i IN ('MORE') DO SET Key=!RANDOM!!RANDOM!!RANDOM!000000000000& ECHO !Key:~0,15!%%i>> %TmpFile%
FOR /F "tokens=*" %%i IN ('TYPE %TmpFile% ^| SORT') DO SET Line=%%i&ECHO.!Line:~15!
DEL %TmpFile%
ENDLOCAL
