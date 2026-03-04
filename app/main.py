"""
main.py — AI 面试辅助外脑 · 主程序入口
==========================================
并发架构总览：

  Qt 主线程（PyQt6 事件循环）
  ├── FloatingWindow（无边框 / 始终置顶 / 半透明）
  │     ├── pyqtSignal: sig_stt(str)    ← 面试官问题文本
  │     ├── pyqtSignal: sig_llm(str)    ← LLM 流式 token / 控制信号
  │     └── pyqtSignal: sig_status(str) ← 状态描述
  │
  └── AsyncWorkerThread（QThread）
        └── asyncio event loop（asyncio.new_event_loop()）
              ├── AudioProcessor.run(stt_q)   ← 音频捕获 + VAD + STT
              ├── _stt_bridge(stt_q, llm_q)   ← 转发 + 触发 UI 信号
              └── process_queries(llm_q, cb)  ← RAG + 搜索 + Gemini 流式

线程安全保证：
  asyncio → Qt：只通过 pyqtSignal.emit()（Qt 保证跨线程 signal 安全）
  Qt → asyncio：只通过 loop.call_soon_threadsafe()（asyncio 保证）

启动参数：
  python main.py              → 正常模式（需要麦克风 / BlackHole）
  python main.py --mock       → 模拟模式（自动注入测试问题，无需麦克风）
  python main.py --mock --debug → 模拟 + 详细日志
"""

# ── 必须在所有 numpy / FAISS / PyTorch 导入之前设置 ────────────────────────────
# macOS 子线程默认栈只有 512KB，OpenBLAS 并行时会在栈上大量分配临时内存导致
# ___chkstk_darwin 触发 SIGBUS/EXC_BAD_ACCESS（栈溢出闪退）。
# 将线程数限制为 1 可完全规避并行内存分配，对推理性能无实质影响（模型本身很小）。
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
# ──────────────────────────────────────────────────────────────────────────────

import asyncio
import importlib
import logging
import sys
from typing import Optional

from PyQt6.QtCore import Qt, QPoint, QThread, pyqtSignal
from PyQt6.QtGui import QFont, QTextCursor, QTextDocument
from PyQt6.QtWidgets import (
    QApplication, QFrame, QHBoxLayout, QLabel,
    QPushButton, QSizeGrip, QSlider, QTextEdit, QListWidget,
    QListWidgetItem, QComboBox, QCheckBox, QSplitter,
    QVBoxLayout, QWidget, QSizePolicy, QDialog, QFormLayout,
    QLineEdit, QDialogButtonBox, QMessageBox, QGroupBox,
)

logger = logging.getLogger("Main")


# ─────────────────────────────────────────────
#  全局样式表（深色半透明面试辅助主题）
# ─────────────────────────────────────────────

STYLESHEET = """
/* ── 主框架 ──────────────────────────────── */
QFrame#mainFrame {
    background-color: rgba(248, 251, 255, 0.94);
    border-radius: 18px;
    border: 1px solid rgba(145, 174, 214, 0.42);
}

/* ── 标题栏 ──────────────────────────────── */
QWidget#titleBar {
    background: qlineargradient(
        x1: 0, y1: 0, x2: 1, y2: 0,
        stop: 0 rgba(241, 247, 255, 0.98),
        stop: 1 rgba(234, 243, 255, 0.98)
    );
    border-radius: 18px 18px 0 0;
    border-bottom: 1px solid rgba(141, 169, 209, 0.45);
}
QLabel#appTitle {
    color: #1A3358;
    font-size: 13px;
    font-weight: 700;
    letter-spacing: 0.4px;
}
QPushButton#wndBtn {
    background: rgba(222, 234, 252, 0.55);
    color: #355B8C;
    border: 1px solid rgba(138, 170, 214, 0.40);
    font-size: 12px;
    border-radius: 6px;
    padding: 2px 6px;
}
QPushButton#wndBtn:hover {
    background: rgba(203, 223, 250, 0.82);
    color: #1F4678;
}

/* ── 内容区域 ─────────────────────────────── */
QWidget#contentArea { background: transparent; }

QLabel#sectionLabel {
    color: #345A89;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 0.5px;
    padding: 1px 0;
}

QSplitter::handle:horizontal {
    background: rgba(148, 175, 214, 0.40);
    width: 2px;
}

/* ── Prompt / 当前问题文本框 ─────────────────── */
QTextEdit#promptDisplay {
    background-color: rgba(255, 255, 255, 0.90);
    color: #1D3658;
    border: 1px solid rgba(148, 177, 219, 0.62);
    border-radius: 12px;
    padding: 10px 12px;
    font-size: 13px;
    selection-background-color: rgba(149, 190, 244, 0.50);
}

QTextEdit#sttDisplay {
    background-color: rgba(248, 252, 255, 0.95);
    color: #193A64;
    border: 1px solid rgba(142, 175, 218, 0.60);
    border-radius: 12px;
    padding: 10px 12px;
    font-size: 13px;
    selection-background-color: rgba(149, 190, 244, 0.55);
}
QTextEdit#sttDisplay QScrollBar:vertical,
QTextEdit#promptDisplay QScrollBar:vertical { width: 0px; }

/* ── AI 回答提示文本框 ─────────────────────── */
QTextEdit#llmDisplay {
    background-color: rgba(245, 255, 250, 0.97);
    color: #0F4C3A;
    border: 1px solid rgba(123, 208, 173, 0.66);
    border-radius: 12px;
    padding: 12px 14px;
    font-size: 14px;
    selection-background-color: rgba(126, 216, 178, 0.55);
}
QTextEdit#llmDisplay QScrollBar:vertical {
    background: rgba(194, 212, 239, 0.48);
    width: 8px;
    border-radius: 4px;
    margin: 0;
}
QTextEdit#llmDisplay QScrollBar::handle:vertical {
    background: rgba(110, 190, 160, 0.78);
    border-radius: 4px;
    min-height: 24px;
}
QTextEdit#llmDisplay QScrollBar::add-line:vertical,
QTextEdit#llmDisplay QScrollBar::sub-line:vertical { height: 0; }

/* ── 实时转录列表 ─────────────────────────── */
QListWidget {
    background-color: rgba(250, 253, 255, 0.97);
    color: #1C3456;
    border: 1px solid rgba(143, 172, 214, 0.58);
    border-radius: 12px;
    padding: 6px 6px;
    outline: none;
    font-size: 14px;
}
QListWidget::item {
    border-radius: 9px;
    padding: 8px 10px;
    margin: 3px 0;
}
QListWidget::item:selected {
    background: rgba(183, 210, 247, 0.66);
    color: #173E70;
}
QListWidget::indicator:unchecked {
    width: 16px;
    height: 16px;
    border: 1px solid rgba(139, 171, 214, 0.78);
    border-radius: 4px;
    background: rgba(255,255,255,0.90);
}
QListWidget::indicator:checked {
    width: 16px;
    height: 16px;
    border: 1px solid rgba(78, 141, 226, 0.95);
    border-radius: 4px;
    background: rgba(104, 166, 247, 0.80);
}

/* ── 控制栏 ──────────────────────────────── */
QWidget#controlBar {
    background: qlineargradient(
        x1: 0, y1: 0, x2: 1, y2: 0,
        stop: 0 rgba(241, 248, 255, 0.96),
        stop: 1 rgba(235, 245, 255, 0.96)
    );
    border-radius: 0 0 18px 18px;
    border-top: 1px solid rgba(145, 172, 212, 0.50);
    padding: 4px 0;
}
QLabel#statusLabel {
    color: #244973;
    font-size: 12px;
    font-weight: 600;
}
QLabel#opacityLabel {
    color: #4A6992;
    font-size: 11px;
    font-weight: 600;
}

/* ── 透明度滑块 ──────────────────────────── */
QSlider#opacitySlider::groove:horizontal {
    height: 6px;
    background: rgba(173, 198, 232, 0.52);
    border-radius: 2px;
}
QSlider#opacitySlider::sub-page:horizontal {
    background: rgba(94, 153, 232, 0.84);
    border-radius: 2px;
}
QSlider#opacitySlider::handle:horizontal {
    width: 14px;
    height: 14px;
    background: rgba(248, 252, 255, 0.98);
    border: 1px solid rgba(106, 146, 202, 0.66);
    border-radius: 7px;
    margin: -4px 0;
}

/* ── 下拉框 / 复选框 / 普通按钮 ───────────────── */
QComboBox {
    background: rgba(255, 255, 255, 0.95);
    color: #1F426C;
    border: 1px solid rgba(136, 168, 212, 0.72);
    border-radius: 10px;
    padding: 5px 10px;
    min-height: 30px;
}
QComboBox::drop-down { border: none; width: 20px; }
QComboBox QAbstractItemView {
    background: rgba(255, 255, 255, 0.98);
    color: #1F426C;
    border: 1px solid rgba(136, 168, 212, 0.72);
    selection-background-color: rgba(183, 210, 247, 0.70);
}
QCheckBox {
    color: #2B4B73;
    spacing: 8px;
    font-size: 12px;
    font-weight: 600;
}
QCheckBox::indicator {
    width: 15px;
    height: 15px;
    border-radius: 4px;
    border: 1px solid rgba(141, 173, 216, 0.75);
    background: rgba(255,255,255,0.92);
}
QCheckBox::indicator:checked {
    background: rgba(100, 163, 245, 0.84);
    border: 1px solid rgba(79, 141, 224, 0.95);
}
QPushButton {
    background-color: rgba(237, 245, 255, 0.95);
    color: #2A4C78;
    border: 1px solid rgba(132, 166, 213, 0.65);
    border-radius: 10px;
    padding: 5px 12px;
    min-height: 30px;
    font-size: 12px;
    font-weight: 600;
}
QPushButton:hover {
    background-color: rgba(221, 236, 254, 0.98);
    border: 1px solid rgba(111, 153, 212, 0.80);
}
QPushButton:pressed {
    background-color: rgba(205, 225, 251, 0.98);
}

/* ── 开始按钮 ────────────────────────────── */
QPushButton#startBtn {
    background-color: rgba(37, 165, 111, 0.93);
    color: #F4FFF9;
    border: 1px solid rgba(55, 152, 114, 0.68);
    border-radius: 10px;
    padding: 6px 16px;
    font-size: 12px;
    font-weight: 700;
    min-height: 32px;
}
QPushButton#startBtn:hover { background-color: rgba(45, 177, 121, 0.96); }
QPushButton#startBtn:pressed { background-color: rgba(33, 142, 98, 0.96); }

/* ── 停止按钮 ────────────────────────────── */
QPushButton#stopBtn {
    background-color: rgba(213, 73, 73, 0.92);
    color: #FFF7F7;
    border: 1px solid rgba(198, 77, 77, 0.70);
    border-radius: 10px;
    padding: 6px 16px;
    font-size: 12px;
    font-weight: 700;
    min-height: 32px;
}
QPushButton#stopBtn:hover { background-color: rgba(228, 81, 81, 0.96); }
QPushButton#stopBtn:pressed { background-color: rgba(190, 61, 61, 0.96); }
"""


# ─────────────────────────────────────────────
#  可拖拽标题栏
# ─────────────────────────────────────────────

class TitleBar(QWidget):
    """
    无边框窗口自定义标题栏。
    - 单击拖拽：移动窗口
    - 双击：切换「始终置顶」
    """

    def __init__(self, parent_window: "FloatingWindow") -> None:
        super().__init__(parent_window)
        self.setObjectName("titleBar")
        self.setFixedHeight(36)
        self._win = parent_window
        self._drag_pos: Optional[QPoint] = None
        self._build()

    def _build(self) -> None:
        row = QHBoxLayout(self)
        row.setContentsMargins(12, 0, 8, 0)
        row.setSpacing(6)

        # 状态指示点（颜色随状态变化）
        self._dot = QLabel("⏺")
        self._dot.setStyleSheet("color: #444; font-size: 10px;")
        row.addWidget(self._dot)

        title = QLabel("AI 面试助手")
        title.setObjectName("appTitle")
        row.addWidget(title)
        row.addStretch()

        # 窗口控制按钮：最小化 / 关闭
        for symbol, slot in [("─", self._win.showMinimized), ("✕", self._win.close)]:
            btn = QPushButton(symbol)
            btn.setObjectName("wndBtn")
            btn.setFixedSize(22, 22)
            btn.clicked.connect(slot)
            row.addWidget(btn)

    def set_dot_color(self, color: str) -> None:
        """更新状态点颜色：#444(灰)/#F5A623(黄)/#4CAF50(绿)/#4A9EFF(蓝)/#E53935(红)"""
        self._dot.setStyleSheet(f"color: {color}; font-size: 10px;")

    # ── 拖拽移动 ──────────────────────────────────────────────────
    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_pos = event.globalPosition().toPoint() - self._win.pos()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if self._drag_pos and event.buttons() == Qt.MouseButton.LeftButton:
            self._win.move(event.globalPosition().toPoint() - self._drag_pos)
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        self._drag_pos = None
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event) -> None:
        """双击标题栏：切换「始终置顶」状态"""
        flags = self._win.windowFlags()
        always_on_top = bool(flags & Qt.WindowType.WindowStaysOnTopHint)
        self._win.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, not always_on_top)
        self._win.show()
        hint = "已取消置顶" if always_on_top else "已设为置顶"
        logger.info(f"窗口置顶状态：{hint}")


class SettingsDialog(QDialog):
    """应用设置弹窗：连接配置、输入设备、模型管理。"""

    PRESET_ENDPOINTS = {
        "OpenRouter": "https://openrouter.ai/api/v1",
        "OpenAI": "https://api.openai.com/v1",
        "SiliconFlow": "https://api.siliconflow.cn/v1",
        "DeepSeek": "https://api.deepseek.com/v1",
        "自定义": "",
    }

    def __init__(self, parent: "FloatingWindow", values: dict, audio_devices: list[tuple[int, str]]) -> None:
        super().__init__(parent)
        self.setWindowTitle("设置")
        self.setModal(True)
        self.resize(700, 560)
        self.setStyleSheet(
            """
            QDialog { background: rgba(247, 252, 255, 0.98); color: #1F426C; }
            QGroupBox {
                border: 1px solid rgba(138, 170, 214, 0.62);
                border-radius: 12px;
                margin-top: 8px;
                padding-top: 10px;
                background: rgba(255,255,255,0.78);
                font-weight: 700;
                color: #2A4E7B;
            }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 6px; }
            QLineEdit, QComboBox, QListWidget {
                background: rgba(255,255,255,0.95);
                color: #1F426C;
                border: 1px solid rgba(136, 168, 212, 0.72);
                border-radius: 10px;
                padding: 6px 8px;
                font-size: 12px;
            }
            QLabel { color: #355B88; font-size: 12px; font-weight: 600; }
            QPushButton {
                background-color: rgba(237, 245, 255, 0.95);
                color: #2A4C78;
                border: 1px solid rgba(132, 166, 213, 0.65);
                border-radius: 10px;
                padding: 6px 12px;
                min-height: 30px;
                font-size: 12px;
                font-weight: 600;
            }
            QPushButton:hover {
                background-color: rgba(221, 236, 254, 0.98);
                border: 1px solid rgba(111, 153, 212, 0.80);
            }
            """
        )
        self._audio_devices = audio_devices
        self._custom_models = list(values.get("custom_models", []))
        self._saved_connection: dict | None = None
        self._saved_audio: dict | None = None
        self._saved_models: dict | None = None

        root = QVBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 12)
        root.setSpacing(12)

        root.addWidget(self._build_connection_group(values))
        root.addWidget(self._build_audio_group(values))
        root.addWidget(self._build_model_group(values), 1)

        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        btns.rejected.connect(self.reject)
        btns.accepted.connect(self.accept)
        close_btn = btns.button(QDialogButtonBox.StandardButton.Close)
        if close_btn is not None:
            close_btn.clicked.connect(self.accept)
        root.addWidget(btns)

    def _build_connection_group(self, values: dict) -> QGroupBox:
        box = QGroupBox("连接设置")
        form = QFormLayout(box)
        form.setContentsMargins(12, 12, 12, 12)
        form.setSpacing(10)

        self._platform = QComboBox()
        for k in self.PRESET_ENDPOINTS.keys():
            self._platform.addItem(k, k)
        self._platform.currentTextChanged.connect(self._on_platform_changed)
        form.addRow("连接方式", self._platform)

        self._base_url = QLineEdit(values.get("base_url", self.PRESET_ENDPOINTS["OpenRouter"]))
        self._base_url.setPlaceholderText("https://openrouter.ai/api/v1")
        form.addRow("API Base URL", self._base_url)

        self._api_key = QLineEdit(values.get("api_key", ""))
        self._api_key.setEchoMode(QLineEdit.EchoMode.Password)
        self._api_key.setPlaceholderText("输入 API Key")
        form.addRow("API Key", self._api_key)

        test_row = QHBoxLayout()
        self._test_model = QLineEdit(values.get("test_model", values.get("active_model", "")))
        self._test_model.setPlaceholderText("测试模型，例如 google/gemini-2.5-pro")
        test_row.addWidget(self._test_model, 1)
        self._test_btn = QPushButton("测试连接")
        self._test_btn.setFixedWidth(110)
        self._test_btn.clicked.connect(self._test_connection)
        test_row.addWidget(self._test_btn)
        form.addRow("连通性测试", test_row)

        conn_ops = QHBoxLayout()
        conn_ops.addStretch()
        self._save_conn_btn = QPushButton("保存连接设置")
        self._save_conn_btn.setFixedWidth(122)
        self._save_conn_btn.clicked.connect(self._save_connection_group)
        conn_ops.addWidget(self._save_conn_btn)
        form.addRow("", conn_ops)

        current_base = self._base_url.text().strip()
        matched = str(values.get("platform", "")).strip() or "自定义"
        if matched not in self.PRESET_ENDPOINTS:
            matched = "自定义"
        if matched == "自定义":
            for name, endpoint in self.PRESET_ENDPOINTS.items():
                if endpoint and endpoint == current_base:
                    matched = name
                    break
        self._platform.setCurrentText(matched)
        return box

    def _build_audio_group(self, values: dict) -> QGroupBox:
        box = QGroupBox("输入设备")
        root = QVBoxLayout(box)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(10)

        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(10)

        self._audio_combo = QComboBox()
        for idx, name in self._audio_devices:
            self._audio_combo.addItem(f"[{idx}] {name}", idx)
        audio_idx = values.get("audio_index")
        pos = self._audio_combo.findData(audio_idx) if isinstance(audio_idx, int) else -1
        self._audio_combo.setCurrentIndex(pos if pos >= 0 else 0)
        row.addWidget(self._audio_combo, 1)
        root.addLayout(row)

        ops = QHBoxLayout()
        ops.addStretch()
        self._save_audio_btn = QPushButton("保存输入设备")
        self._save_audio_btn.setFixedWidth(118)
        self._save_audio_btn.clicked.connect(self._save_audio_group)
        ops.addWidget(self._save_audio_btn)
        root.addLayout(ops)
        return box

    def _build_model_group(self, values: dict) -> QGroupBox:
        box = QGroupBox("模型管理")
        root = QVBoxLayout(box)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(10)

        input_row = QHBoxLayout()
        self._new_model = QLineEdit()
        self._new_model.setPlaceholderText("输入模型 ID，例如 google/gemini-3.1-pro-preview")
        input_row.addWidget(self._new_model, 1)
        self._update_btn = QPushButton("Update")
        self._update_btn.setFixedWidth(94)
        self._update_btn.clicked.connect(self._add_model_from_input)
        input_row.addWidget(self._update_btn)
        root.addLayout(input_row)

        self._models_list = QListWidget()
        self._models_list.setSelectionMode(QListWidget.SelectionMode.SingleSelection)
        self._models_list.setMinimumHeight(150)
        root.addWidget(self._models_list, 1)
        self._refresh_models_list()

        ops = QHBoxLayout()
        ops.addStretch()
        self._save_models_btn = QPushButton("保存模型管理")
        self._save_models_btn.setFixedWidth(122)
        self._save_models_btn.clicked.connect(self._save_models_group)
        ops.addWidget(self._save_models_btn)
        self._del_model_btn = QPushButton("删除选中模型")
        self._del_model_btn.setFixedWidth(140)
        self._del_model_btn.clicked.connect(self._delete_selected_model)
        ops.addWidget(self._del_model_btn)
        root.addLayout(ops)
        return box

    def _on_platform_changed(self, label: str) -> None:
        endpoint = self.PRESET_ENDPOINTS.get(label, "")
        if endpoint:
            self._base_url.setText(endpoint)

    def _refresh_models_list(self) -> None:
        self._models_list.clear()
        for m in self._custom_models:
            self._models_list.addItem(m)

    def _add_model_from_input(self) -> None:
        model = self._new_model.text().strip()
        if not model:
            QMessageBox.information(self, "提示", "请输入模型名称")
            return
        if model in self._custom_models:
            QMessageBox.information(self, "提示", "该模型已存在")
            return
        self._custom_models.append(model)
        self._custom_models.sort()
        self._refresh_models_list()
        self._new_model.clear()

    def _delete_selected_model(self) -> None:
        item = self._models_list.currentItem()
        if item is None:
            QMessageBox.information(self, "提示", "请先选择要删除的模型")
            return
        model = item.text().strip()
        if not model:
            return
        ok = QMessageBox.question(
            self,
            "确认删除",
            f"确定删除模型：\n{model}",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if ok != QMessageBox.StandardButton.Yes:
            return
        self._custom_models = [m for m in self._custom_models if m != model]
        self._refresh_models_list()

    def _test_connection(self) -> None:
        api_key = self._api_key.text().strip()
        base_url = self._base_url.text().strip()
        test_model = self._test_model.text().strip()
        if not test_model:
            current = self._models_list.currentItem()
            if current is not None:
                test_model = current.text().strip()
        if not test_model:
            QMessageBox.warning(self, "测试失败", "请输入测试模型或先选中一个模型")
            return
        self._test_btn.setEnabled(False)
        self._test_btn.setText("测试中...")
        try:
            from llm_engine import probe_connection
            ok, msg = asyncio.run(probe_connection(api_key, base_url, test_model))
            if ok:
                QMessageBox.information(self, "测试通过", f"连接可用。\n{msg}")
            else:
                QMessageBox.critical(self, "测试失败", msg)
        except Exception as e:
            QMessageBox.critical(self, "测试失败", f"{type(e).__name__}: {e}")
        finally:
            self._test_btn.setEnabled(True)
            self._test_btn.setText("测试连接")

    def _save_connection_group(self) -> None:
        self._saved_connection = {
            "platform": self._platform.currentData(),
            "base_url": self._base_url.text().strip(),
            "api_key": self._api_key.text().strip(),
        }
        QMessageBox.information(self, "已保存", "连接设置已保存。")

    def _save_audio_group(self) -> None:
        self._saved_audio = {
            "audio_index": self._audio_combo.currentData(),
            "audio_text": self._audio_combo.currentText(),
        }
        QMessageBox.information(self, "已保存", "输入设备设置已保存。")

    def _save_models_group(self) -> None:
        self._saved_models = {"custom_models": list(self._custom_models)}
        QMessageBox.information(self, "已保存", "模型管理设置已保存。")

    def get_values(self) -> dict:
        return {
            "saved_connection": self._saved_connection,
            "saved_audio": self._saved_audio,
            "saved_models": self._saved_models,
        }


# ─────────────────────────────────────────────
#  后台异步工作线程
# ─────────────────────────────────────────────

class AsyncWorkerThread(QThread):
    """
    在独立线程中运行单一 asyncio 事件循环。

    设计原则：
      - QThread 只负责持有事件循环，不处理任何业务逻辑
      - 所有业务协程（音频、STT、LLM）通过 asyncio.gather 并发运行
      - 与 UI 的唯一通信方式是 pyqtSignal（线程安全，自动排队投递到 Qt 主线程）
    """

    sig_stt    = pyqtSignal(str)   # STT 识别的面试官问题
    sig_llm    = pyqtSignal(str)   # LLM token / 控制信号（CLEAR / DONE / ERROR）
    sig_status = pyqtSignal(str)   # 状态更新字符串
    sig_error  = pyqtSignal(str)   # 错误消息

    def __init__(self, mock_mode: bool = False, parent=None) -> None:
        super().__init__(parent)
        self._mock = mock_mode
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._audio_proc = None   # AudioProcessor 实例（正常模式）
        self._llm_q: Optional[asyncio.Queue] = None
        from config import AUTO_ANSWER_ENABLED
        self._auto_answer = AUTO_ANSWER_ENABLED
        self._audio_device_index: Optional[int] = None
        # 将子线程栈扩展到 64MB（macOS 默认仅 512KB），防止 FAISS/numpy 在线程池中栈溢出
        self.setStackSize(64 * 1024 * 1024)

    # ── QThread 入口 ──────────────────────────────────────────────

    def run(self) -> None:
        """在独立线程中创建并运行 asyncio 事件循环"""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._async_main())
        except asyncio.CancelledError:
            logger.info("所有后台任务已取消（正常关闭）")
        except Exception as e:
            logger.exception("AsyncWorkerThread 发生未捕获异常")
            self.sig_error.emit(str(e))
        finally:
            self._drain_loop()

    def _drain_loop(self) -> None:
        """清理事件循环中残余未完成的任务"""
        if not self._loop or self._loop.is_closed():
            return
        pending = asyncio.all_tasks(self._loop)
        if pending:
            self._loop.run_until_complete(
                asyncio.gather(*pending, return_exceptions=True)
            )
        try:
            from llm_engine import close_client
            self._loop.run_until_complete(close_client())
        except Exception:
            pass
        self._loop.close()
        logger.info("asyncio 事件循环已关闭")

    # ── 主异步任务 ────────────────────────────────────────────────

    async def _async_main(self) -> None:
        """协调所有后台协程（正常模式 or 模拟模式）"""

        # ① 初始化 RAG（耗时，用线程池避免阻塞 asyncio loop）
        from config import RAG_ENABLED, RAG_FORCE_REBUILD
        self.sig_status.emit("🟡 初始化知识库...")
        try:
            rag_mod = importlib.import_module("rag_search")
            init_rag_fast = getattr(rag_mod, "init_rag_fast")
            await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: init_rag_fast(
                    force_rebuild=RAG_FORCE_REBUILD,
                    progress_cb=lambda msg: self.sig_status.emit(f"🟡 {msg}"),
                ),
            )
            if RAG_ENABLED:
                logger.info("RAG 知识库初始化流程结束")
            else:
                logger.info("RAG 默认关闭，已跳过初始化")
        except Exception as e:
            logger.warning(f"RAG 初始化失败（{e}），将自动回退纯 LLM")

        self.sig_status.emit("🟢 监听中...")

        # ② 选择运行模式
        if self._mock:
            await self._mock_mode()
        else:
            await self._normal_mode()

    async def _normal_mode(self) -> None:
        """正常模式：真实音频 + STT + LLM"""
        from llm_engine import process_queries

        try:
            from audio_processor import AudioProcessor
        except Exception as e:
            raise RuntimeError(
                f"音频模块加载失败（通常是 sounddevice/torch/portaudio 缺失）: {e}"
            ) from e

        stt_q: asyncio.Queue[str] = asyncio.Queue()
        llm_q: asyncio.Queue = asyncio.Queue()
        self._llm_q = llm_q

        # STT 桥接：转发文本 + 触发 UI 信号
        async def _stt_bridge() -> None:
            while True:
                text = await stt_q.get()
                self.sig_stt.emit(text)      # → 更新面试官问题显示
                if self._auto_answer:
                    await llm_q.put((text, 0))    # 自动转录低优先级

        # LLM token 回调（pyqtSignal.emit 线程安全）
        def _llm_cb(token: str) -> None:
            self.sig_llm.emit(token)

        try:
            self._audio_proc = AudioProcessor(device_index=self._audio_device_index)
        except Exception as e:
            raise RuntimeError(f"音频处理器初始化失败: {e}") from e

        # 三个核心协程并发运行
        await asyncio.gather(
            self._audio_proc.run(stt_q),   # 音频捕获 + VAD + STT
            _stt_bridge(),                  # STT → UI 桥接
            process_queries(llm_q, _llm_cb),  # RAG + 搜索 + Gemini
        )

    async def _mock_mode(self) -> None:
        """
        模拟模式（--mock）：无需麦克风，自动注入测试问题。
        用于验证 UI + LLM 流程，不依赖音频设备。
        """
        from llm_engine import process_queries

        MOCK_QUESTIONS = [
            "请介绍一下你自己，以及你最近在做的主要工作。",
            "你在 AI 和大模型领域有什么经验？可以结合项目来谈谈。",
            "你怎么看待 AI 基础设施赛道的竞争格局？",
            "Tell me about a challenging project you've worked on recently.",
        ]

        llm_q: asyncio.Queue = asyncio.Queue()

        def _llm_cb(token: str) -> None:
            self.sig_llm.emit(token)

        async def _mock_injector() -> None:
            """每隔 12 秒注入一个测试问题"""
            await asyncio.sleep(2)    # 等待 RAG 初始化完成
            for q in MOCK_QUESTIONS:
                logger.info(f"[模拟] 注入测试问题: {q[:40]!r}")
                self.sig_stt.emit(q)
                await llm_q.put((q, 0))
                await asyncio.sleep(14)

        await asyncio.gather(
            _mock_injector(),
            process_queries(llm_q, _llm_cb),
        )

    # ── 停止接口（从 Qt 主线程调用）──────────────────────────────

    def request_stop(self) -> None:
        """线程安全地请求停止所有后台任务（不阻塞调用方）"""
        if self._loop and not self._loop.is_closed():
            self._loop.call_soon_threadsafe(self._cancel_all)

    def set_auto_answer(self, enabled: bool) -> None:
        self._auto_answer = enabled

    def submit_manual_query(self, text: str) -> None:
        if not text.strip():
            return
        if self._loop and not self._loop.is_closed():
            self._loop.call_soon_threadsafe(self._enqueue_manual_query, text)

    def _enqueue_manual_query(self, text: str) -> None:
        if self._llm_q is None:
            return
        try:
            self._llm_q.put_nowait((text.strip(), 10))  # 手动提问高优先级
        except asyncio.QueueFull:
            logger.warning("LLM 队列已满，忽略本次手动提问")

    def set_model(self, model: str) -> None:
        try:
            from llm_engine import set_runtime_model
            set_runtime_model(model)
        except Exception as e:
            logger.warning("切换模型失败: %s", e)

    def set_language_profile(self, profile: str) -> None:
        try:
            from audio_processor import set_runtime_language_profile
            set_runtime_language_profile(profile)
        except Exception as e:
            logger.warning("切换语言策略失败: %s", e)

    def set_audio_device(self, index: Optional[int]) -> None:
        self._audio_device_index = index

    def _cancel_all(self) -> None:
        """在 asyncio 线程中取消所有任务（含音频流）"""
        if self._audio_proc:
            self._audio_proc.stop()
        self._llm_q = None
        try:
            rag_mod = importlib.import_module("rag_search")
            cancel_rag_build = getattr(rag_mod, "cancel_rag_build", None)
            if callable(cancel_rag_build):
                cancel_rag_build()
        except Exception:
            pass
        for task in asyncio.all_tasks(self._loop):
            task.cancel()


# ─────────────────────────────────────────────
#  手动提问线程（停止监听后仍可用）
# ─────────────────────────────────────────────
class ManualQueryThread(QThread):
    sig_llm = pyqtSignal(str)
    sig_error = pyqtSignal(str)

    def __init__(self, question: str, parent=None) -> None:
        super().__init__(parent)
        self._question = question

    def run(self) -> None:
        import asyncio
        from llm_engine import handle_single_query, close_client

        async def _run_once() -> None:
            await handle_single_query(self._question, self.sig_llm.emit)
            await close_client()

        try:
            asyncio.run(_run_once())
        except Exception as e:
            self.sig_error.emit(str(e))


# ─────────────────────────────────────────────
#  悬浮窗主界面
# ─────────────────────────────────────────────

class FloatingWindow(QWidget):
    """
    无边框 / 半透明悬浮窗（默认不置顶，可手动切换置顶）。

    UI 布局：
    ┌─────────────────────────────────────┐
    │ ⏺ AI 面试助手              ─    ✕  │  ← TitleBar（可拖拽）
    ├─────────────────────────────────────┤
    │ 🎤 面试官问题                        │
    │ ┌─────────────────────────────────┐ │
    │ │  STT 识别文字（只读）            │ │
    │ └─────────────────────────────────┘ │
    │ 🤖 AI 回答提示                       │
    │ ┌─────────────────────────────────┐ │
    │ │  • 流式生成要点（实时追加）       │ │
    │ └─────────────────────────────────┘ │
    ├─────────────────────────────────────┤
    │ 🟢 监听中...  [透明度─]  [▶开始]   │  ← 控制栏
    └─────────────────────────────────────┘                                       ╗ resize
    """

    def __init__(self, mock_mode: bool = False) -> None:
        super().__init__()
        self._mock = mock_mode
        self._worker: Optional[AsyncWorkerThread] = None
        self._manual_thread: Optional[ManualQueryThread] = None
        self._last_done_answer: str = ""
        self._answer_snapshot: str = ""
        self._current_answer_qid: str = ""
        self._running = False
        self._build_window()
        self._build_ui()
        self._apply_extra_prompt()
        self._position_to_top_right()

    # ── 窗口初始化 ────────────────────────────────────────────────

    def _build_window(self) -> None:
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setMinimumSize(760, 360)
        self.resize(920, 520)
        self.setWindowTitle("AI 面试助手")

    def _position_to_top_right(self) -> None:
        screen = QApplication.primaryScreen().availableGeometry()
        self.move(screen.width() - self.width() - 20, 40)

    # ── UI 构建 ───────────────────────────────────────────────────

    def _build_ui(self) -> None:
        # 外层：1px margin 为将来阴影预留空间
        outer = QVBoxLayout(self)
        outer.setContentsMargins(1, 1, 1, 1)
        outer.setSpacing(0)

        # 主框架（深色背景 + 圆角）
        frame = QFrame()
        frame.setObjectName("mainFrame")
        frame.setStyleSheet(STYLESHEET)
        outer.addWidget(frame)

        inner = QVBoxLayout(frame)
        inner.setContentsMargins(0, 0, 0, 0)
        inner.setSpacing(0)

        # 标题栏
        self._title_bar = TitleBar(self)
        inner.addWidget(self._title_bar)

        # 内容区
        content = QWidget()
        content.setObjectName("contentArea")
        cl = QVBoxLayout(content)
        cl.setContentsMargins(12, 8, 12, 6)
        cl.setSpacing(5)

        split = QSplitter(Qt.Orientation.Horizontal)
        self._content_splitter = split

        # 左：当前问题 + AI 回答
        left = QWidget()
        ll = QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 0, 0)
        ll.setSpacing(5)
        ll.addWidget(self._section_label("🧠  Prompt 策略（可编辑）"))

        self._prompt_box = QTextEdit()
        self._prompt_box.setObjectName("promptDisplay")
        self._prompt_box.setPlaceholderText(
            "在这里输入你想强调的回答风格/行业重点/角色定位，例如："
            "“偏战略投资视角，强调商业闭环、竞争壁垒、财务可行性与执行路径。”"
        )
        self._prompt_box.setFixedHeight(92)
        ll.addWidget(self._prompt_box)

        prompt_ops = QHBoxLayout()
        prompt_ops.setContentsMargins(0, 0, 0, 0)
        prompt_ops.setSpacing(6)
        self._apply_prompt_btn = QPushButton("应用 Prompt")
        self._apply_prompt_btn.setFixedHeight(26)
        self._apply_prompt_btn.clicked.connect(self._apply_extra_prompt)
        prompt_ops.addWidget(self._apply_prompt_btn)
        prompt_ops.addStretch()
        ll.addLayout(prompt_ops)

        ll.addWidget(self._section_label("🎤  当前问题"))

        self._stt_box = QTextEdit()
        self._stt_box.setObjectName("sttDisplay")
        self._stt_box.setReadOnly(True)
        self._stt_box.setFixedHeight(82)
        self._stt_box.setPlaceholderText("等待面试官提问...")
        ll.addWidget(self._stt_box)

        ll.addWidget(self._section_label("🤖  AI 回答提示"))
        self._llm_box = QTextEdit()
        self._llm_box.setObjectName("llmDisplay")
        self._llm_box.setReadOnly(True)
        self._llm_box.setPlaceholderText("AI 回答将在此显示...")
        ll.addWidget(self._llm_box, 1)

        # 右：大尺寸实时转录历史
        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.setSpacing(5)
        rl.addWidget(self._section_label("📝  实时转录（可滚动，勾选后手动提问）"))
        self._transcript_list = QListWidget()
        self._transcript_list.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        self._transcript_list.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._transcript_list.setTextElideMode(Qt.TextElideMode.ElideNone)
        self._transcript_list.setWordWrap(True)
        self._transcript_list.setUniformItemSizes(False)
        self._transcript_list.setSpacing(3)
        rl.addWidget(self._transcript_list, 1)

        right_ops = QHBoxLayout()
        right_ops.setContentsMargins(0, 0, 0, 0)
        right_ops.setSpacing(6)
        self._ask_checked_btn = QPushButton("提问勾选")
        self._ask_checked_btn.setFixedHeight(28)
        self._ask_checked_btn.clicked.connect(self._ask_checked_transcripts)
        right_ops.addWidget(self._ask_checked_btn)

        self._ctx_btn = QPushButton("更新上下文")
        self._ctx_btn.setFixedHeight(28)
        self._ctx_btn.clicked.connect(self._update_transcript_context)
        right_ops.addWidget(self._ctx_btn)
        rl.addLayout(right_ops)

        split.addWidget(left)
        split.addWidget(right)
        split.setSizes([420, 300])
        split.splitterMoved.connect(lambda *_: self._refresh_all_transcripts_layout())
        cl.addWidget(split, 1)

        inner.addWidget(content, 1)
        inner.addWidget(self._build_control_bar())

        # 右下角 resize grip
        grip_wrap = QHBoxLayout()
        grip_wrap.setContentsMargins(0, 0, 4, 2)
        grip_wrap.addStretch()
        grip = QSizeGrip(frame)
        grip.setStyleSheet("background: transparent;")
        grip_wrap.addWidget(
            grip, 0,
            Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignRight
        )
        inner.addLayout(grip_wrap)

        # 字体：中英文混排优化
        font = QFont()
        font.setFamilies([
            "Avenir Next", "PingFang SC",
            "Hiragino Sans GB", "Microsoft YaHei", "Helvetica Neue"
        ])
        self._stt_box.setFont(font)
        self._llm_box.setFont(font)
        self._transcript_list.setFont(font)
        self._prompt_box.setFont(font)

        # 默认 Prompt（可直接编辑覆盖）
        default_prompt = (
            "你是我的面试实时智囊。请优先给出：结论、依据、取舍、风险、落地动作。"
            "多用可验证事实和项目证据，避免空话。若问题涉及行业判断，给出短中长期影响。"
        )
        self._prompt_box.setPlainText(default_prompt)

    def _wrap_text_to_width(self, text: str, max_width: int) -> str:
        """按像素宽度硬换行，保证每条转录不超出右侧面板宽度。"""
        if not text or max_width <= 20:
            return text
        fm = self._transcript_list.fontMetrics()
        lines: list[str] = []
        for para in text.splitlines() or [""]:
            # 英文优先按单词换行，中文按字符换行，避免单词被切断。
            if " " in para:
                tokens = para.split(" ")
                current = ""
                for tok in tokens:
                    candidate = tok if not current else f"{current} {tok}"
                    if fm.horizontalAdvance(candidate) <= max_width:
                        current = candidate
                    else:
                        if current:
                            lines.append(current)
                        if fm.horizontalAdvance(tok) <= max_width:
                            current = tok
                        else:
                            # 超长单词回退字符级切分
                            current = ""
                            tmp = ""
                            for ch in tok:
                                c2 = tmp + ch
                                if fm.horizontalAdvance(c2) <= max_width:
                                    tmp = c2
                                else:
                                    if tmp:
                                        lines.append(tmp)
                                    tmp = ch
                            current = tmp
                if current:
                    lines.append(current)
            else:
                current = ""
                for ch in para:
                    candidate = current + ch
                    if fm.horizontalAdvance(candidate) <= max_width:
                        current = candidate
                    else:
                        if current:
                            lines.append(current)
                        current = ch
                lines.append(current)
        return "\n".join(lines)

    def _refresh_transcript_item_layout(self, item: QListWidgetItem) -> None:
        raw = str(item.data(Qt.ItemDataRole.UserRole) or item.text())
        # 始终显示完整原文：不做省略，不做硬截断
        item.setText(raw)

        max_w = max(120, self._transcript_list.viewport().width() - 42)
        doc = QTextDocument()
        doc.setDefaultFont(self._transcript_list.font())
        doc.setPlainText(raw)
        doc.setTextWidth(float(max_w))

        # 叠加 item padding 与留白，防止最后一行被裁剪
        text_h = int(doc.size().height())
        item_h = max(26, text_h + 16)
        from PyQt6.QtCore import QSize
        item.setSizeHint(QSize(max_w, item_h))

    def _refresh_all_transcripts_layout(self) -> None:
        for i in range(self._transcript_list.count()):
            self._refresh_transcript_item_layout(self._transcript_list.item(i))

    def _list_audio_inputs(self) -> list[tuple[int, str]]:
        try:
            import sounddevice as sd
            result: list[tuple[int, str]] = []
            for i, dev in enumerate(sd.query_devices()):
                if dev.get("max_input_channels", 0) >= 1:
                    result.append((i, str(dev.get("name", f"Device {i}"))))
            return result
        except Exception:
            return []

    def _env_file_path(self):
        from pathlib import Path
        return Path(__file__).resolve().parent / ".env"

    def _read_env_map(self) -> dict[str, str]:
        env_path = self._env_file_path()
        data: dict[str, str] = {}
        if not env_path.exists():
            return data
        for line in env_path.read_text(encoding="utf-8").splitlines():
            s = line.strip()
            if not s or s.startswith("#") or "=" not in s:
                continue
            k, v = s.split("=", 1)
            data[k.strip()] = v.strip()
        return data

    def _write_env_updates(self, updates: dict[str, str]) -> None:
        env_path = self._env_file_path()
        lines = env_path.read_text(encoding="utf-8").splitlines() if env_path.exists() else []
        out: list[str] = []
        consumed: set[str] = set()
        for line in lines:
            if "=" in line and not line.lstrip().startswith("#"):
                k = line.split("=", 1)[0].strip()
                if k in updates:
                    out.append(f"{k}={updates[k]}")
                    consumed.add(k)
                    continue
            out.append(line)
        for k, v in updates.items():
            if k not in consumed:
                out.append(f"{k}={v}")
        env_path.write_text("\n".join(out) + "\n", encoding="utf-8")

    def _refresh_model_combo(self, preferred: Optional[str] = None) -> None:
        from llm_engine import get_available_models, get_runtime_model
        current = preferred or self._model_combo.currentText() or get_runtime_model()
        models = get_available_models()
        self._model_combo.blockSignals(True)
        self._model_combo.clear()
        self._model_combo.addItems(models)
        idx = self._model_combo.findText(current)
        if idx < 0:
            idx = self._model_combo.findText(get_runtime_model())
        self._model_combo.setCurrentIndex(idx if idx >= 0 else 0)
        self._model_combo.blockSignals(False)

    @staticmethod
    def _section_label(text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setObjectName("sectionLabel")
        return lbl

    def _build_control_bar(self) -> QWidget:
        bar = QWidget()
        bar.setObjectName("controlBar")
        bar.setMinimumHeight(92)
        outer = QVBoxLayout(bar)
        outer.setContentsMargins(16, 8, 16, 10)
        outer.setSpacing(8)

        row_top = QHBoxLayout()
        row_top.setContentsMargins(0, 0, 0, 0)
        row_top.setSpacing(10)

        row_bottom = QHBoxLayout()
        row_bottom.setContentsMargins(0, 0, 0, 0)
        row_bottom.setSpacing(10)

        self._status_lbl = QLabel("⚪ 就绪")
        self._status_lbl.setObjectName("statusLabel")
        row_top.addWidget(self._status_lbl)

        from llm_engine import get_available_models, get_runtime_model
        self._model_combo = QComboBox()
        self._model_combo.setMinimumSize(210, 32)
        self._model_combo.setMaximumWidth(380)
        self._model_combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._model_combo.addItems(get_available_models())
        current_model = get_runtime_model()
        idx = self._model_combo.findText(current_model)
        if idx >= 0:
            self._model_combo.setCurrentIndex(idx)
        self._model_combo.currentTextChanged.connect(self._on_model_changed)
        row_top.addWidget(self._model_combo, 2)

        from config import STT_LANGUAGE_PROFILE
        self._lang_combo = QComboBox()
        self._lang_combo.setMinimumSize(118, 32)
        self._lang_combo.setMaximumWidth(150)
        self._lang_combo.addItem("中文面试", "zh")
        self._lang_combo.addItem("英文面试", "en")
        self._lang_combo.addItem("自动双语", "auto")
        pidx = self._lang_combo.findData(STT_LANGUAGE_PROFILE if STT_LANGUAGE_PROFILE in ("zh", "en", "auto") else "zh")
        self._lang_combo.setCurrentIndex(pidx if pidx >= 0 else 0)
        row_top.addWidget(self._lang_combo)
        self._lang_combo.setFixedHeight(34)
        self._lang_apply_btn = QPushButton("应用语言")
        self._lang_apply_btn.setFixedSize(104, 34)
        self._lang_apply_btn.clicked.connect(self._apply_language_profile)
        row_top.addWidget(self._lang_apply_btn)

        from config import AUTO_ANSWER_ENABLED
        self._auto_answer_chk = QCheckBox("自动答")
        self._auto_answer_chk.setFixedHeight(30)
        self._auto_answer_chk.setChecked(AUTO_ANSWER_ENABLED)
        self._auto_answer_chk.toggled.connect(self._on_auto_answer_toggled)
        row_top.addWidget(self._auto_answer_chk)
        row_top.addStretch(1)

        self._settings_btn = QPushButton("设置")
        self._settings_btn.setFixedSize(88, 34)
        self._settings_btn.clicked.connect(self._open_settings_dialog)
        row_bottom.addWidget(self._settings_btn)
        row_bottom.addStretch(1)

        # 透明度滑块
        opacity_lbl = QLabel("透明")
        opacity_lbl.setObjectName("opacityLabel")
        row_bottom.addWidget(opacity_lbl)

        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setObjectName("opacitySlider")
        self._slider.setRange(30, 100)
        self._slider.setValue(92)
        self._slider.setFixedWidth(96)
        self._slider.valueChanged.connect(lambda v: self.setWindowOpacity(v / 100))
        row_bottom.addWidget(self._slider)

        # 开始 / 停止按钮
        self._btn = QPushButton("▶  开始监听")
        self._btn.setObjectName("startBtn")
        self._btn.setMinimumWidth(146)
        self._btn.setFixedHeight(34)
        self._btn.clicked.connect(self._toggle)
        row_bottom.addWidget(self._btn)

        outer.addLayout(row_top)
        outer.addLayout(row_bottom)

        # 初始不透明度
        self.setWindowOpacity(0.92)
        return bar

    # ── 后台线程控制 ──────────────────────────────────────────────

    def _toggle(self) -> None:
        if self._running:
            self._stop_worker()
        else:
            self._start_worker()

    def _start_worker(self) -> None:
        self._worker = AsyncWorkerThread(mock_mode=self._mock, parent=self)
        self._worker.set_auto_answer(self._auto_answer_chk.isChecked())
        self._worker.set_model(self._model_combo.currentText())
        lang_profile = self._lang_combo.currentData()
        if isinstance(lang_profile, str):
            self._worker.set_language_profile(lang_profile)
        from config import AUDIO_DEVICE_INDEX
        device_index = AUDIO_DEVICE_INDEX
        if device_index is None:
            env_map = self._read_env_map()
            raw = env_map.get("AUDIO_DEVICE_INDEX", "").strip()
            if raw.isdigit():
                device_index = int(raw)
        if isinstance(device_index, int) and device_index >= 0:
            self._worker.set_audio_device(device_index)
        self._apply_extra_prompt()
        self._worker.sig_stt.connect(self._on_stt)
        self._worker.sig_llm.connect(self._on_llm_token)
        self._worker.sig_status.connect(self._on_status)
        self._worker.sig_error.connect(self._on_error)
        self._worker.finished.connect(self._on_worker_finished)
        self._worker.start()

        self._running = True
        self._set_btn_stop()
        self._on_status("🟡 初始化中...")
        self._title_bar.set_dot_color("#F5A623")

        if self._mock:
            self._llm_box.setPlaceholderText("【模拟模式】测试问题将每 12s 自动注入...")

    def _stop_worker(self) -> None:
        if self._worker:
            self._worker.request_stop()
            self._worker.quit()
            if not self._worker.wait(5000):      # 最多等 5s
                logger.warning("线程超时，强制终止")
                self._worker.terminate()
                self._worker.wait()
            self._worker = None

        self._running = False
        self._set_btn_start()
        self._on_status("⚪ 已停止（可勾选转录手动提问）")
        self._title_bar.set_dot_color("#444")

    def _set_btn_start(self) -> None:
        self._btn.setObjectName("startBtn")
        self._btn.setText("▶  开始监听")
        self._btn.style().polish(self._btn)

    def _set_btn_stop(self) -> None:
        self._btn.setObjectName("stopBtn")
        self._btn.setText("⏹  停止监听")
        self._btn.style().polish(self._btn)

    # ── 信号槽（均在 Qt 主线程中执行，线程安全）─────────────────

    def _on_stt(self, text: str) -> None:
        """显示面试官问题"""
        self._stt_box.setPlainText(text)
        if text.strip():
            from config import TRANSCRIPT_MAX_ITEMS
            item = QListWidgetItem(text.strip())
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Unchecked)
            item.setData(Qt.ItemDataRole.UserRole, text.strip())
            self._refresh_transcript_item_layout(item)
            self._transcript_list.addItem(item)
            if self._transcript_list.count() > TRANSCRIPT_MAX_ITEMS:
                self._transcript_list.takeItem(0)
            self._transcript_list.scrollToBottom()

    def _ask_checked_transcripts(self) -> None:
        picked: list[str] = []
        for i in range(self._transcript_list.count()):
            item = self._transcript_list.item(i)
            if item.checkState() == Qt.CheckState.Checked:
                text = (item.data(Qt.ItemDataRole.UserRole) or item.text()).strip()
                if text:
                    picked.append(text)
        if not picked:
            self._on_status("⚪ 请先勾选转录条目")
            return
        # 勾选后自动取消勾选，避免重复提交
        for i in range(self._transcript_list.count()):
            item = self._transcript_list.item(i)
            if item.checkState() == Qt.CheckState.Checked:
                item.setCheckState(Qt.CheckState.Unchecked)
        query = "\n".join(picked)
        self._submit_query(query)

    def _submit_query(self, text: str) -> None:
        question = (text or "").strip()
        if not question:
            return
        self._stt_box.setPlainText(question)
        self._on_status("🔵 生成中...")
        self._title_bar.set_dot_color("#4A9EFF")
        if self._worker and self._running:
            self._worker.submit_manual_query(question)
            return

        if self._manual_thread and self._manual_thread.isRunning():
            self._on_status("⚪ 上一条手动提问仍在处理中")
            return
        self._manual_thread = ManualQueryThread(question, parent=self)
        self._manual_thread.sig_llm.connect(self._on_llm_token)
        self._manual_thread.sig_error.connect(self._on_error)
        self._manual_thread.finished.connect(self._on_manual_finished)
        self._manual_thread.start()

    def _on_manual_finished(self) -> None:
        if not self._running:
            self._on_status("⚪ 已停止（可继续手动提问）")
            self._title_bar.set_dot_color("#444")

    def _update_transcript_context(self) -> None:
        lines: list[str] = []
        for i in range(self._transcript_list.count()):
            item = self._transcript_list.item(i)
            text = (item.data(Qt.ItemDataRole.UserRole) or item.text()).strip()
            if text:
                lines.append(text)
        from llm_engine import set_transcript_context
        payload = "\n".join(lines)
        set_transcript_context(payload)
        self._on_status(f"⚪ 已更新上下文（{len(lines)} 条）")

    def _on_model_changed(self, model: str) -> None:
        if self._worker and self._running:
            self._worker.set_model(model)
        self._on_status(f"⚪ 模型: {model}")

    def _apply_audio_device(self, index: int, text: str) -> None:
        if not isinstance(index, int) or index < 0:
            self._on_status("⚪ 无可用音频输入设备")
            return
        name = text.split("]", 1)[1].strip() if "]" in text else text
        try:
            self._write_env_updates({
                "AUDIO_DEVICE_INDEX": str(index),
                "AUDIO_DEVICE_NAME": name,
            })
        except Exception as e:
            logger.warning("写入 .env 音频配置失败: %s", e)
        if self._worker and self._running:
            self._on_status(f"⚪ 已选择音频 {text}（停止后重新开始生效）")
        else:
            self._on_status(f"⚪ 已选择音频 {text}")

    def _open_settings_dialog(self) -> None:
        env_map = self._read_env_map()
        from config import AUDIO_DEVICE_INDEX
        audio_idx = AUDIO_DEVICE_INDEX
        if audio_idx is None:
            raw = env_map.get("AUDIO_DEVICE_INDEX", "").strip()
            if raw.isdigit():
                audio_idx = int(raw)
        values = {
            "platform": env_map.get("LLM_PROVIDER", "OpenRouter"),
            "base_url": env_map.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
            "api_key": env_map.get("OPENROUTER_API_KEY", ""),
            "audio_index": audio_idx,
            "custom_models": [x.strip() for x in env_map.get("OPENROUTER_CUSTOM_MODELS", "").split(",") if x.strip()],
            "active_model": self._model_combo.currentText().strip(),
        }
        devices = self._list_audio_inputs()
        if not devices:
            self._on_status("⚪ 当前无可用输入设备")
            return
        dlg = SettingsDialog(self, values, devices)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        v = dlg.get_values()
        saved_connection = v.get("saved_connection") or {}
        saved_audio = v.get("saved_audio") or {}
        saved_models = v.get("saved_models") or {}
        changed = False

        if saved_connection:
            api_key = saved_connection.get("api_key", "")
            base_url = saved_connection.get("base_url", "") or "https://openrouter.ai/api/v1"
            platform = saved_connection.get("platform", "OpenRouter")
            try:
                self._write_env_updates({
                    "LLM_PROVIDER": str(platform),
                    "OPENROUTER_BASE_URL": str(base_url),
                    "OPENROUTER_API_KEY": str(api_key),
                })
                from llm_engine import set_runtime_connection
                set_runtime_connection(api_key=str(api_key), base_url=str(base_url))
                changed = True
            except Exception as e:
                logger.warning("保存连接设置失败: %s", e)

        if saved_models:
            custom_models = saved_models.get("custom_models", [])
            try:
                self._write_env_updates({
                    "OPENROUTER_CUSTOM_MODELS": ",".join(custom_models),
                })
                from llm_engine import set_runtime_custom_models, get_runtime_model
                set_runtime_custom_models(custom_models)
                preferred = get_runtime_model()
                self._refresh_model_combo(preferred=preferred)
                if preferred not in [self._model_combo.itemText(i) for i in range(self._model_combo.count())]:
                    self._model_combo.setCurrentText(self._model_combo.itemText(0))
                changed = True
            except Exception as e:
                logger.warning("保存模型设置失败: %s", e)

        if saved_audio:
            audio_index = saved_audio.get("audio_index")
            audio_text = saved_audio.get("audio_text", "")
            if isinstance(audio_index, int) and audio_index >= 0:
                self._apply_audio_device(audio_index, str(audio_text))
                changed = True

        if changed:
            self._on_status("⚪ 设置已保存并生效")
        else:
            self._on_status("⚪ 未检测到已保存的设置变更")

    def _apply_extra_prompt(self) -> None:
        from llm_engine import set_extra_prompt
        text = self._prompt_box.toPlainText().strip()
        set_extra_prompt(text)
        self._on_status("⚪ 已应用 Prompt 策略")

    def _apply_language_profile(self) -> None:
        profile = self._lang_combo.currentData()
        if not isinstance(profile, str):
            return
        whisper_lang = "" if profile == "auto" else profile
        try:
            from pathlib import Path
            env_path = Path(__file__).resolve().parent / ".env"
            lines = env_path.read_text(encoding="utf-8").splitlines() if env_path.exists() else []
            out: list[str] = []
            updated = {"STT_LANGUAGE_PROFILE": False, "WHISPER_LANGUAGE": False}
            for line in lines:
                if line.startswith("STT_LANGUAGE_PROFILE="):
                    out.append(f"STT_LANGUAGE_PROFILE={profile}")
                    updated["STT_LANGUAGE_PROFILE"] = True
                elif line.startswith("WHISPER_LANGUAGE="):
                    out.append(f"WHISPER_LANGUAGE={whisper_lang}")
                    updated["WHISPER_LANGUAGE"] = True
                else:
                    out.append(line)
            if not updated["STT_LANGUAGE_PROFILE"]:
                out.append(f"STT_LANGUAGE_PROFILE={profile}")
            if not updated["WHISPER_LANGUAGE"]:
                out.append(f"WHISPER_LANGUAGE={whisper_lang}")
            env_path.write_text("\n".join(out) + "\n", encoding="utf-8")
        except Exception as e:
            logger.warning("写入 .env 语言配置失败: %s", e)
        if self._worker and self._running:
            self._worker.set_language_profile(profile)
            self._on_status(f"⚪ 语言策略已生效: {profile}")
        else:
            self._on_status(f"⚪ 语言策略已保存: {profile}")

    def _on_auto_answer_toggled(self, enabled: bool) -> None:
        if self._worker and self._running:
            self._worker.set_auto_answer(enabled)
        mode = "自动回答" if enabled else "手动勾选提问"
        self._on_status(f"⚪ 模式: {mode}")

    def _on_llm_token(self, token: str) -> None:
        """
        处理 LLM 流式 token / 控制信号。

        信号协议（来自 llm_engine.py）：
          "<<<CLEAR>>>" → 清空文本框，新答案开始
          "<<<DONE>>>"  → 流式结束，状态恢复监听
          "<<<ERROR>>>" → 发生错误
          其他字符串    → 追加到文本框（流式 token）
        """
        from llm_engine import CLEAR_SIGNAL, DONE_SIGNAL, ERROR_SIGNAL, CANCELED_SIGNAL

        if token.startswith(f"{CLEAR_SIGNAL}:"):
            qid = token.split(":", 1)[1].strip()
            self._answer_snapshot = self._llm_box.toPlainText()
            if qid and qid != self._current_answer_qid:
                # 仅新问题清空；同问题重试/补流不清空，避免“回答闪没”
                self._current_answer_qid = qid
                self._llm_box.clear()
            self._on_status("🔵 生成中...")
            self._title_bar.set_dot_color("#4A9EFF")
        elif token == CLEAR_SIGNAL:
            # 兼容旧信号协议
            self._answer_snapshot = self._llm_box.toPlainText()
            self._llm_box.clear()
            self._on_status("🔵 生成中...")
            self._title_bar.set_dot_color("#4A9EFF")

        elif token == DONE_SIGNAL:
            self._last_done_answer = self._llm_box.toPlainText()
            logger.info("UI 回答框长度=%d", len(self._last_done_answer.strip()))
            if self._running:
                self._on_status("🟢 监听中...")
                self._title_bar.set_dot_color("#4CAF50")
            else:
                self._on_status("⚪ 已停止（可勾选转录手动提问）")
                self._title_bar.set_dot_color("#444")
        elif token == CANCELED_SIGNAL:
            rollback = self._answer_snapshot or self._last_done_answer
            if rollback.strip():
                self._llm_box.setPlainText(rollback)
            if self._running:
                self._on_status("🟢 监听中...")
                self._title_bar.set_dot_color("#4CAF50")
            else:
                self._on_status("⚪ 已停止（可勾选转录手动提问）")
                self._title_bar.set_dot_color("#444")

        elif token == ERROR_SIGNAL:
            self._llm_box.append("\n⚠  生成失败，请检查 .env 中的 API Key 和网络连接")
            if self._running:
                self._on_status("🟢 监听中...")
                self._title_bar.set_dot_color("#4CAF50")
            else:
                self._on_status("⚪ 已停止（可勾选转录手动提问）")
                self._title_bar.set_dot_color("#444")
        elif token.startswith(f"{ERROR_SIGNAL}:"):
            detail = token.split(":", 1)[1].strip() or "未知错误"
            self._llm_box.append(f"\n⚠  生成失败：{detail}")
            if self._running:
                self._on_status("🟢 监听中...")
                self._title_bar.set_dot_color("#4CAF50")
            else:
                self._on_status("⚪ 已停止（可勾选转录手动提问）")
                self._title_bar.set_dot_color("#444")

        else:
            # 流式追加：光标移到末尾再插入，保持自动滚动
            self._llm_box.moveCursor(QTextCursor.MoveOperation.End)
            self._llm_box.insertPlainText(token)
            self._llm_box.ensureCursorVisible()

    def _on_status(self, msg: str) -> None:
        self._status_lbl.setText(msg)
        # 同步打印到终端日志，便于在 command 中观察 RAG / STT 实时进度
        try:
            if getattr(self, "_last_status_log", None) != msg:
                logger.info("Status — %s", msg)
                self._last_status_log = msg
        except Exception:
            pass

    def _on_error(self, msg: str) -> None:
        self._llm_box.setPlainText(f"⚠  系统错误：\n{msg}")
        self._on_status("🔴 错误")
        self._title_bar.set_dot_color("#E53935")
        logger.error(f"Worker 错误: {msg}")

    def _on_worker_finished(self) -> None:
        """线程结束（可能是正常停止，也可能是意外退出）"""
        if self._running:
            # 意外退出（非用户主动停止）
            logger.warning("后台线程意外退出")
            self._running = False
            self._set_btn_start()
            self._on_status("⚪ 已停止（意外退出）")
            self._title_bar.set_dot_color("#444")

    # ── 快捷键 / 窗口事件 ─────────────────────────────────────────

    def keyPressEvent(self, event) -> None:
        key = event.key()
        if key == Qt.Key.Key_Space:
            self._toggle()
        elif key in (Qt.Key.Key_Escape, Qt.Key.Key_Q):
            self.close()
        else:
            super().keyPressEvent(event)

    def closeEvent(self, event) -> None:
        """关闭前确保后台线程干净退出"""
        if self._running:
            self._stop_worker()
        if self._manual_thread and self._manual_thread.isRunning():
            self._manual_thread.quit()
            if not self._manual_thread.wait(2000):
                self._manual_thread.terminate()
                self._manual_thread.wait()
        event.accept()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._refresh_all_transcripts_layout()


# ─────────────────────────────────────────────
#  主入口
# ─────────────────────────────────────────────

def main(mock_mode: bool = False) -> None:
    # macOS 高 DPI 支持
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )

    app = QApplication(sys.argv)
    app.setApplicationName("AI 面试助手")
    app.setQuitOnLastWindowClosed(True)

    window = FloatingWindow(mock_mode=mock_mode)
    window.show()

    if mock_mode:
        logger.info("=" * 50)
        logger.info("  模拟模式启动（--mock）")
        logger.info("  无需麦克风，自动注入测试问题")
        logger.info("  点击「开始监听」查看 AI 回答效果")
        logger.info("=" * 50)

    sys.exit(app.exec())


if __name__ == "__main__":
    import argparse
    import os
    from pathlib import Path

    # ── macOS PyQt6 平台插件路径自动修正 ──────────────────────────────
    # 当 PyQt6 安装在 conda/venv 中时，Qt 找不到 cocoa 插件。
    # 通过自动探测 site-packages 路径并设置环境变量来解决。
    def _fix_qt_plugin_path() -> None:
        try:
            import PyQt6
            pyqt_dir = Path(PyQt6.__file__).parent
            plugin_path = pyqt_dir / "Qt6" / "plugins" / "platforms"
            if plugin_path.exists():
                current = os.environ.get("QT_QPA_PLATFORM_PLUGIN_PATH", "")
                if str(plugin_path) not in current:
                    os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = str(plugin_path)
        except Exception:
            pass  # 已有系统 Qt 安装时忽略

    _fix_qt_plugin_path()
    # ─────────────────────────────────────────────────────────────────

    parser = argparse.ArgumentParser(description="AI 面试辅助外脑")
    parser.add_argument(
        "--mock", action="store_true",
        help="模拟模式：自动注入测试问题，无需麦克风（用于 UI + LLM 调试）"
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="开启 DEBUG 级别日志"
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )

    main(mock_mode=args.mock)
