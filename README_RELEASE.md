# Interview Assistant - 可迁移发布版

本目录是独立发布工程，不依赖原始开发目录中的绝对路径和私密配置。

## 目录说明
- `app/`：应用源码（PyQt + STT + LLM）
- `app/.env.example`：配置模板（不含密钥）
- `data/docs`：可选 RAG 文档目录
- `data/index`：RAG 向量索引目录
- `start.command` / `start.bat`：一键启动
- `scripts/build_macos_dmg.sh`：构建 macOS DMG
- `scripts/build_windows_exe.ps1`：构建 Windows 包
- `.github/workflows/build-release.yml`：CI 自动构建（exe+dmg）

## 本地启动
```bash
cd InterviewAssistant
./start.command
```

## 打包
### macOS
```bash
cd InterviewAssistant
bash scripts/build_macos_dmg.sh
```

### Windows（在 Windows 机器或 GitHub Actions）
```powershell
cd Application/InterviewAssistant
./scripts/build_windows_exe.ps1
```

## 发布前敏感信息扫描
```bash
cd InterviewAssistant
python scripts/sanitize_check.py
```
