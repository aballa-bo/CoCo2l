import sys
import traceback
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QTimer
from coco2 import MainWindow

app = QApplication(sys.argv)
win = MainWindow()
win._analyze._raw_edit.setText("img/DSC0136.NEF")
win._analyze._raw_path = "img/DSC0136.NEF"
win._analyze._output_dir.setText("output")

# Uncheck devignetting explicitly
win._analyze._devignetting.setChecked(False)

def start_run():
    try:
        win._analyze._run()
        print(f"Args used: {win._analyze._proc.arguments()}")
    except Exception as e:
        traceback.print_exc()
        app.quit()

def check_done():
    if win._analyze._proc is not None and win._analyze._proc.state() == win._analyze._proc.ProcessState.NotRunning:
        print("Done. Log:", win._analyze._log.toPlainText())
        app.quit()

QTimer.singleShot(0, start_run)
timer = QTimer()
timer.timeout.connect(check_done)
timer.start(100)

sys.exit(app.exec())
