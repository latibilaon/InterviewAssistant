# InterviewAssistant

## 项目目的 | Project Purpose

### 中文
InterviewAssistant 是一个面向真实面试场景的实时辅助工具，目标是让用户在不同设备上都能快速完成部署并稳定使用。

核心目的：
1. 实时转录面试问题（中文/英文）。
2. 基于已配置模型生成结构化回答建议。
3. 通过设置界面完成 API、模型与音频设备配置，降低上手门槛。
4. 支持发布与迁移（macOS / Windows），便于分发和复用。

### English
InterviewAssistant is a portable real-time interview assistant designed for practical interview scenarios and cross-device deployment.

Core goals:
1. Real-time transcription for interview audio (Chinese/English).
2. Structured answer guidance from configured LLM models.
3. In-app configuration for API, model, and audio input to reduce setup friction.
4. Portable packaging and distribution for macOS and Windows.

---

## 操作方法 | How To Use

### A. 本地启动 | Local Run

#### macOS
```bash
cd InterviewAssistant
./start.command
```

#### Windows
```bat
cd InterviewAssistant
start.bat
```

说明 / Notes:
1. 首次启动会自动创建 `.venv` 并安装依赖。  
   First run creates `.venv` and installs dependencies automatically.
2. 启动后进入主界面，点击底部“设置”完成连接配置。  
   After launch, open **Settings** to configure connection.

### B. 设置 API 与模型 | Configure API and Models

#### 中文
1. 打开“设置” -> “连接设置”。
2. 选择连接方式（OpenRouter / OpenAI / 自定义）。
3. 填写 `API Base URL` 与 `API Key`。
4. 在“连通性测试”输入模型并点击“测试连接”。
5. 测试通过后点击“保存连接设置”。
6. 在“模型管理”输入模型 ID，点击 `Update` 加入列表，再点击“保存模型管理”。
7. 在主界面模型下拉框切换当前模型。

#### English
1. Open **Settings** -> **Connection Settings**.
2. Select provider (OpenRouter / OpenAI / Custom).
3. Fill in `API Base URL` and `API Key`.
4. Enter a model in **Connection Test** and click **Test Connection**.
5. Click **Save Connection Settings** after success.
6. In **Model Management**, input model ID, click `Update`, then click **Save Model Management**.
7. Switch active model from the model dropdown in the main UI.

### C. 输入设备设置 | Audio Input Setup

#### 中文
1. 打开“设置” -> “输入设备”。
2. 选择正确的输入设备。
3. 点击“保存输入设备”。
4. 若已在监听中，建议停止后重新开始监听。

#### English
1. Open **Settings** -> **Input Device**.
2. Select the correct audio input.
3. Click **Save Input Device**.
4. If listening is already running, stop and start listening again.

### D. 打包发布 | Packaging

#### macOS DMG
```bash
cd InterviewAssistant
bash scripts/build_macos_dmg.sh
```
Output: `release/InterviewAssistant-macos.dmg`

#### Windows Package (PowerShell)
```powershell
cd InterviewAssistant
./scripts/build_windows_exe.ps1
```
Output: `release/InterviewAssistant-windows.zip`

### E. GitHub 自动构建 | GitHub Actions Build

```bash
git tag v1.0.0
git push origin v1.0.0
```

中文：推送 tag 后，进入 GitHub Actions 下载构建产物。  
English: After pushing a tag, download build artifacts from GitHub Actions.
