import os, sys
from pathlib import Path
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
sys.path.insert(0,str(Path('Project_AI/NeuroLab').resolve()))
from PySide6.QtWidgets import QApplication
from PySide6.QtTest import QTest
from app.ui.main_window import MainWindow
from PySide6.QtGui import QFontDatabase, QFont
app=QApplication([])
for font in ["segoeui.ttf", "segoeuib.ttf", "seguisym.ttf"]:
    QFontDatabase.addApplicationFont("C:/Windows/Fonts/"+font)
app.setFont(QFont("Segoe UI",10))
window=MainWindow()
window.show()
app.processEvents()
for name in window.pages:
    window._switch(name)
    QTest.qWait(200)
    app.processEvents()
    window.grab().save('outputs/design_review/'+name.lower().replace(' ','_').replace('?','')+'.png')
    assert window.stack.currentWidget() is window.page_containers[name]
for name in ['Prediction','Training','Dataset','Settings','Dashboard']:
    window._switch(name)
    QTest.qWait(20)
window.resize(1100,720)
QTest.qWait(200)
window._switch('Training')
QTest.qWait(200)
window.grab().save('outputs/design_review/training_compact.png')
prediction = window.pages['Prediction']
prediction.btn_demo.click()
assert prediction.class_lbl.text() in __import__('app.ml.dataset',fromlist=['FASHION_MNIST_CLASSES']).FASHION_MNIST_CLASSES
assert prediction.bars_layout.count() == 5
training = window.pages['Training']
training.combo_mode.setCurrentIndex(1)
assert training.spin_epochs.value() == 5 and training.spin_batch.value() == 128
training.combo_mode.setCurrentIndex(0)
assert training.spin_epochs.value() == 3 and training.spin_batch.value() == 64
assert window.pages['Training'].model is window.pages['Prediction'].model
window.close()
print('PASS: 9 pages constructed, navigation, rapid transitions, compact resize and close')
