@echo off
REM Build script for FINDER C++ on Windows

echo ======================================
echo FINDER C++ Build Script
echo ======================================
echo.

REM Check if build directory exists
if exist build (
    echo Cleaning old build directory...
    rmdir /s /q build
)

echo Creating build directory...
mkdir build
cd build

echo.
echo Configuring with CMake...
cmake ..

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ERROR: CMake configuration failed!
    echo Make sure you have CMake installed and in your PATH.
    pause
    exit /b 1
)

echo.
echo Building Release configuration...
cmake --build . --config Release

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ERROR: Build failed!
    pause
    exit /b 1
)

echo.
echo ======================================
echo Build successful!
echo ======================================
echo.
echo Executable: build\Release\finder.exe
echo.
echo To run clustering:
echo   cd build\Release
echo   finder.exe input.csv output.csv [2D^|3D]
echo.
echo Example:
echo   finder.exe ..\..\examples\sample2d.csv results.csv
echo.
echo For help:
echo   finder.exe --help
echo.
pause
