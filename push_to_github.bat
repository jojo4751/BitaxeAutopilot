@echo off
echo 🚀 BitAxe V2.0.0 GitHub Push Script
echo =====================================

cd /d "C:\DevPacks\BITAXE_V2.0.0"

echo 📋 Checking git status...
git status

echo.
echo 📤 Adding remote repository...
git remote add origin https://github.com/jojo4751/BitaxeAutopilot.git 2>nul

echo.
echo 🌟 Pushing master branch...
git push -u origin master

echo.
echo 🏷️ Pushing version tag...
git push origin v2.0.0

echo.
echo ✅ Push complete! Check: https://github.com/jojo4751/BitaxeAutopilot
pause