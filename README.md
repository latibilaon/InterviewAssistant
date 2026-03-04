# InterviewAssistant

一个可迁移的 AI 面试助手发布工程（中英双语说明）。
A portable release project for an AI Interview Assistant (bilingual guide).

## 1) 项目用途 | What This Repo Is
- 用于在不同设备快速部署并运行面试助手。
- Includes runtime app source, startup scripts, packaging scripts, and CI workflow.
- 默认不包含真实密钥，不绑定本地绝对路径。

## 2) 目录结构 | Structure
- `app/`：主程序源码（PyQt + STT + LLM + 可选 RAG）
- `app/.env.example`：配置模板（复制为 `.env`）
- `data/docs/`：可选 RAG 文档目录
- `data/index/`：RAG 索引缓存目录
- `start.command`：macOS 一键启动
- `start.bat`：Windows 一键启动
- `scripts/build_macos_dmg.sh`：构建 macOS DMG
- `scripts/build_windows_exe.ps1`：构建 Windows 包
- `.github/workflows/build-release.yml`：GitHub Actions 自动构建

## 3) 首次启动 | First Run
### macOS
```bash
cd InterviewAssistant
./start.command
```

### Windows
```bat
cd InterviewAssistant
start.bat
```

首次会自动创建虚拟环境并安装依赖。
On first run, it will create `.venv` and install dependencies.

## 4) API 与模型配置 | API & Model Setup
推荐在应用内设置页配置（无需手改文件）：
Use the in-app **Settings** page:

1. 打开 `设置` -> `连接设置`  
2. 选择连接方式（OpenRouter/OpenAI/自定义）  
3. 填写 `Base URL`、`API Key`、测试模型  
4. 点击 `测试连接`（成功后会弹窗）  
5. 点击 `保存连接设置`  
6. 在 `模型管理` 输入模型，点 `Update`，再点 `保存模型管理`

可选文件配置：复制 `app/.env.example` 为 `app/.env` 后填写。

## 5) 打包 | Packaging
### macOS DMG
```bash
cd InterviewAssistant
bash scripts/build_macos_dmg.sh
```
输出：`release/InterviewAssistant-macos.dmg`

### Windows package (PowerShell)
```powershell
cd InterviewAssistant
./scripts/build_windows_exe.ps1
```
输出：`release/InterviewAssistant-windows.zip`

## 6) GitHub 自动构建 | GitHub Actions Build
推送 tag 即可触发：
```bash
git tag v1.0.0
git push origin v1.0.0
```

然后在 `Actions` 下载构建产物（Artifacts）。
Then download artifacts from the `Actions` page.

## 7) 发布前安全检查 | Pre-release Security Check
```bash
python scripts/sanitize_check.py
```
用于检查明显敏感信息（token、绝对路径等）。

## 8) 常见问题 | Common Issues
- `AuthenticationError`：API Key 无效或未配置。
- push 被拒绝 workflow scope：你的 GitHub PAT 缺少 `workflow` 权限。
- 音频无输入：在设置页保存正确输入设备，重启监听。

## 9) 建议工作流 | Recommended Workflow
1. 在本地调好设置并测试通过。  
2. 提交代码并打新 tag。  
3. 等 Actions 构建完成后下载安装包分发。
