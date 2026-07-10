#!/bin/bash
# 一键上传脚本
# 用法: ./push.sh "你修改了什么"
# 如果不写描述，默认用 "update"

MSG="${1:-update}"

echo "📦 暂存所有修改..."
git add --all

echo "✏️  提交: $MSG"
git commit -m "$MSG"

echo "🚀 推送到 GitHub..."
git push

echo "✅ 完成!"
