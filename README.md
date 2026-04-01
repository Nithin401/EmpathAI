
# EmpathAI – Emotion-Aware Conversational Chatbot

EmpathAI is an intelligent conversational chatbot designed to understand human emotions and respond with empathy. It analyzes user input using Natural Language Processing (NLP) techniques and generates context-aware, emotionally appropriate responses.

---

## Features

* Real-time conversational chatbot
* Emotion detection from user input (happy, sad, angry, etc.)
* Context-aware and empathetic responses
* Clean and interactive web interface
* Mood tracking (planned feature)
* Chat history integration (Firebase-ready)

---

## Tech Stack

**Frontend:**

* HTML
* CSS
* JavaScript

**Backend:**

* Python
* Flask

**AI / NLP:**

* Text preprocessing
* Emotion classification (rule-based / ML-based)

**Database (Optional):**

* Firebase (for chat storage and analytics)

---

## Project Structure

```
EmpathAI/
│
├── templates/
│   ├── index.html
│   ├── style.css
│   └── script.js
│
├── app.py
├── requirements.txt
└── README.md
```

---

## Installation and Setup

1. Clone the repository:

```
git clone https://github.com/your-username/empathai.git
cd empathai
```

2. Create a virtual environment:

```
python -m venv venv
venv\Scripts\activate   # Windows
source venv/bin/activate  # Mac/Linux
```

3. Install dependencies:

```
pip install -r requirements.txt
```

4. Run the Flask application:

```
python app.py
```

5. Open in browser:

```
http://127.0.0.1:5000
```

---

## How It Works

1. The user enters a message in the chat interface
2. The backend processes the input using NLP techniques
3. The system detects the emotion (e.g., sad, happy, stressed)
4. A suitable empathetic response is generated
5. The response is displayed in real time

---

## Future Improvements

* Integration of advanced models such as BERT or Transformers
* Mood analytics dashboard
* Mobile application using Flutter
* AWS Lambda integration for serverless processing
* Voice-based interaction

---

## Contributing

Contributions are welcome. You can fork the repository and submit a pull request.

---

## License

This project is open-source and available under the MIT License.

---

## Author

Nithin Varma Nampally
LinkedIn: https://www.linkedin.com/in/nithin-varma-nampally-71a327287/
GitHub: https://github.com/Nithin401

---

## Support

If you find this project useful, consider giving it a star on GitHub.
