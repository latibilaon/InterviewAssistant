# 设置 API 与模型（设置界面优先）

## 推荐方式：设置界面
1. 启动应用后点击底部 `设置`。
2. 在 `连接设置` 中填写：
   - 连接方式（OpenRouter / OpenAI / 自定义）
   - API Base URL
   - API Key
3. 在 `连通性测试` 输入模型后点击 `测试连接`。
4. 测试通过后点击 `保存连接设置`。
5. 在 `模型管理` 输入模型 ID，点击 `Update` 加入列表，再点 `保存模型管理`。
6. 关闭设置，在主界面模型下拉切换模型。

## 文件方式（可选）
编辑 `app/.env`：
- `OPENROUTER_API_KEY=`
- `OPENROUTER_BASE_URL=`
- `OPENROUTER_MODEL=`
- `OPENROUTER_CUSTOM_MODELS=`

修改后重启应用。
