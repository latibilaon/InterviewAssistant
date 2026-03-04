# Interview Assistant - 可迁移发布版

本目录是独立发布工程，不依赖原始开发目录中的绝对路径和私密配置。

## 目录说明
- `app/`：应用源码（PyQt + STT + LLM）
- `app/.env.example`：配置模板（不含密钥）
- `data/docs`：可选 RAG 文档目录
- `data/index`：RAG 向量索引目录
- `start.command` / `start.bat`：一键启动
- `scripts/build_macos_app.sh`：仅构建 macOS `.app`
- `scripts/package_macos_dmg.sh`：将已构建 `.app` 打包成 DMG
- `scripts/build_macos_dmg.sh`：完整流程（build app + package dmg）
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
# 步骤1：只构建 .app
bash scripts/build_macos_app.sh

# 步骤2：只打包 dmg（不重复构建）
bash scripts/package_macos_dmg.sh
```

或一键完整流程：
```bash
bash scripts/build_macos_dmg.sh
```

### Windows（在 Windows 机器或 GitHub Actions）
```powershell
cd Application/InterviewAssistant
./scripts/build_windows_exe.ps1
```

## GitHub Actions 触发建议
- `workflow_dispatch` 时可选 `target`：
  - `windows`：只跑 Windows（不跑 macOS / DMG）
  - `macos`：只跑 macOS（build app + package dmg）
  - `all`：全部运行

## 发布前敏感信息扫描
```bash
cd InterviewAssistant
python scripts/sanitize_check.py
```
