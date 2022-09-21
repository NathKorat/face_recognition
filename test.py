from PySide6.QtCore import *
from PySide6.QtGui import *
from PySide6.QtWidgets import *
import sys


def acess_camera():
    import camera

app = QApplication(sys.argv)
window = QWidget()
window.resize(1000,500)
font = QFont('Arail', 30)

layout = QVBoxLayout()
layout.setAlignment(Qt.AlignTop |Qt.AlignCenter)

label = QLabel("Menu")
label.setFont(font)

button = QPushButton("Edit/Add Datasets")
button.setFont(font)
button1 = QPushButton("Start")

button1.clicked.connect(acess_camera)
button1.clicked.connect(sys.exit)

layout.addWidget(label)
layout.addWidget(button)
layout.addWidget(button1)

window.setLayout(layout)

window.show()
sys.exit(app.exec())
