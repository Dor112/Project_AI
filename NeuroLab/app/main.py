#!/usr/bin/env python3
import os, sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

def main():
    from PySide6.QtWidgets import QApplication
    from PySide6.QtGui import QFont
    app = QApplication(sys.argv)
    app.setApplicationName("NeuroLab")
    app.setFont(QFont("Segoe UI", 10))
    from app.ui.main_window import MainWindow
    window = MainWindow()
    window.show()
    return app.exec()

if __name__ == "__main__":
    sys.exit(main())