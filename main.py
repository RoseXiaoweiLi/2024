import sys
from tkinter import messagebox

from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QLineEdit, QPushButton, QTextEdit, \
    QDesktopWidget, QHBoxLayout
import requests
import joblib

model = joblib.load('phishingemaildetection_model.pkl')
vectorizer = joblib.load('phishingemaildetection_vectorizer.pkl')

class ChatWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.layout = QVBoxLayout()
        self.text_original = QTextEdit()
        self.text_original.setFixedSize(1000, 200)
        self.text_original.setPlaceholderText("Enter the original email content")
        self.check_button = QPushButton("Check")
        self.check_button.clicked.connect(self.predict)
        self.check_button.setFixedSize(100,40)
        self.check_box = QTextEdit()
        self.check_box.setReadOnly(True)
        self.check_box.setFixedSize(1000,20)
        self.text_input = QTextEdit()
        self.text_input.setFixedSize(1000,200)
        self.text_input.setPlaceholderText("Enter the change guidance")
        self.send_button = QPushButton("Modify")
        self.send_button.clicked.connect(self.send_message)
        self.send_button.setFixedSize(100,40)
        self.message_box = QTextEdit()
        self.message_box.setReadOnly(True)
        self.layout.addWidget(self.text_original)
        self.layout.addWidget(self.check_button)
        self.layout.addWidget(self.check_box)
        self.layout.addWidget(self.text_input)
        self.layout.addWidget(self.send_button)
        self.layout.addWidget(self.message_box)
        self.setLayout(self.layout)
        self.setGeometry(800, 800, 1000, 1000)
        self.setWindowTitle("Modify the email with LLM")
        self.centerWindow()
        self.show()

    def centerWindow(self):
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    def predict(self):
        originalText = self.text_original.toPlainText()
        print(originalText)
        new_text_transformed = vectorizer.transform([originalText])

        # Make predictions
        prediction = model.predict(new_text_transformed)
        print(prediction)
        if prediction == 0:
            self.check_box.setText("It is a safe email.")
        else:
            self.check_box.setText("It is a phishing email.")
        #return prediction

    def send_message(self):
        message = self.text_input.toPlainText()
        print(message)
        url = 'http://localhost:8080/v1/chat/completions'
        headers = {
            'accept': 'application/json',
            'Content-Type': 'application/json'
        }
        dataEn = {
            'messages': [
                {
                    'content': 'You are a helpful assistant.',
                    'role': 'system'
                },
                {
                    'content': message,
                    'role': 'user'
                }
            ]
        }

        resp = requests.post(
            #url="http://localhost:8080/completion",
            #json={"prompt": message},
            #headers={"Content-Type": "application/json;charset=utf-8"}
            url, headers=headers, json=dataEn
        )
        #response = resp.json()["content"]
        response = resp.json()['choices'][0]['message']['content']
        self.message_box.setText(self.message_box.toPlainText() + "\nYou: " + message + "\nGPT: " + response)
        self.text_input.setText("")


def main():
    app = QApplication(sys.argv)
    window = ChatWindow()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()