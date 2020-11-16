import sys
from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow
from PyQt5.QtCore import Qt

# Encapsulate main window in a class
class MainWindow(QMainWindow):
    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)

        # Set window title
        self.setWindowTitle("Intelligent Stock Trader")
        self.setMinimumSize(1024, 512)

        # "Hello World!" label
        label = QLabel("Hello World!")
        label.setAlignment(Qt.AlignCenter)
        self.setCentralWidget(label) # Make label take up entire window

# Create application.
app = QApplication(sys.argv) # sys.argv are commandline arguments passed in when the program runs.

# Create main window.
window = MainWindow()
window.show()

# Start the main program loop.
print("Program starting.")
app.exec_()

# Program terminating.
print("Program termininating.")