import sys
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QTimer
from coco2 import MainWindow
import traceback

app = QApplication(sys.argv)
win = MainWindow()
win._analyze._raw_edit.setText("img/DSC0136.NEF")
win._analyze._raw_path = "img/DSC0136.NEF"
win._analyze._output_dir.setText("output")
win._analyze._run_btn.setEnabled(True)

def simulate_click():
    print("Clicking run button...")
    try:
        win._analyze._run_btn.click()
    except Exception as e:
        print("Crash during click!")
        traceback.print_exc()
        app.quit()

def check_done():
    if win._analyze._proc is not None and win._analyze._proc.state() == win._analyze._proc.ProcessState.NotRunning:
        print("Process finished.")
        app.quit()

QTimer.singleShot(100, simulate_click)
timer = QTimer()
timer.timeout.connect(check_done)
timer.start(100)

sys.exit(app.exec())
