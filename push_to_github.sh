#!/bin/bash
echo "🚀 BitAxe V2.0.0 GitHub Push Script"
echo "====================================="

cd "$(dirname "$0")"

echo "📋 Checking git status..."
git status

echo ""
echo "📤 Adding remote repository..."
git remote add origin https://github.com/jojo4751/BitaxeAutopilot.git 2>/dev/null || true

echo ""
echo "🌟 Pushing main branch..."
git push -u origin main

echo ""
echo "🏷️ Pushing version tag..."
git push origin v2.0.0

echo ""
echo "✅ Push complete! Check: https://github.com/jojo4751/BitaxeAutopilot"
echo "Press any key to continue..."
read -n 1