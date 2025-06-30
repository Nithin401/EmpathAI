import os
import json
import uuid
from flask import Flask, request, jsonify, render_template, redirect, url_for
from flask_cors import CORS
from emotion_analysis import analyze_emotion
import requests
from datetime import datetime
from dotenv import load_dotenv
import boto3
from botocore.exceptions import ClientError
from moto import mock_aws
import firebase_admin
from firebase_admin import credentials, firestore
import logging
from PIL import Image
import io

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Set a custom Transformers cache folder
os.environ['TRANSFORMERS_CACHE'] = 'D:/lightweight_hf_cache'

# Initialize Flask app
app = Flask(__name__, template_folder='templates', static_folder='static')
CORS(app)

# Storage Configuration
S3_BUCKET_NAME = os.getenv('S3_BUCKET_NAME', 'empathai-chat-sessions')
STORAGE_BACKEND = os.getenv('STORAGE_BACKEND', 's3')

# Initialize Mock S3 client
@mock_aws
def initialize_s3_client():
    s3_client = boto3.client('s3')
    try:
        s3_client.head_bucket(Bucket=S3_BUCKET_NAME)
        logger.info(f"Connected to mock S3 bucket '{S3_BUCKET_NAME}'")
    except ClientError as e:
        if e.response['Error']['Code'] == '404':
            logger.info(f"Creating mock S3 bucket '{S3_BUCKET_NAME}'")
            s3_client.create_bucket(Bucket=S3_BUCKET_NAME)
        else:
            logger.error(f"Failed to initialize mock S3: {e}")
            raise
    return s3_client

s3_client = initialize_s3_client()

# Initialize Firebase
try:
    cred = credentials.Certificate(os.getenv("FIREBASE_CREDENTIALS_PATH", "serviceAccountKey.json"))
    firebase_admin.initialize_app(cred)
    db = firestore.client()
    logger.info("Firebase initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize Firebase: {str(e)}")
    raise

CHAT_COLLECTION = 'chat_sessions'

def save_chat_session(chat_id, messages, deleted=False):
    if STORAGE_BACKEND == 'firestore':
        save_chat_session_to_firestore(chat_id, messages, deleted)
    else:
        save_chat_session_to_s3(chat_id, messages, deleted)

def load_all_chat_sessions():
    if STORAGE_BACKEND == 'firestore':
        return load_all_chat_sessions_from_firestore()
    return load_all_chat_sessions_from_s3()

def delete_chat_session(chat_id):
    if STORAGE_BACKEND == 'firestore':
        return delete_chat_session_from_firestore(chat_id)
    return delete_chat_session_from_s3(chat_id)

def restore_chat_session(chat_id):
    if STORAGE_BACKEND == 'firestore':
        return restore_chat_session_from_firestore(chat_id)
    return restore_chat_session_from_s3(chat_id)

@mock_aws
def save_chat_session_to_s3(chat_id, messages, deleted=False):
    try:
        messages_json = json.dumps(messages, ensure_ascii=False)
        created_at = messages[0].get('timestamp', datetime.now().isoformat()) if messages else datetime.now().isoformat()
        prefix = "deleted_chat_sessions/" if deleted else "chat_sessions/"
        s3_client.put_object(
            Bucket=S3_BUCKET_NAME,
            Key=f"{prefix}{chat_id}.json",
            Body=json.dumps({'chat_id': chat_id, 'messages': messages, 'created_at': created_at, 'deleted': deleted}, ensure_ascii=False)
        )
        logger.info(f"Saved chat session {chat_id} to mock S3 (deleted={deleted})")
    except Exception as e:
        logger.error(f"Error saving chat session {chat_id} to mock S3: {str(e)}")

def save_chat_session_to_firestore(chat_id, messages, deleted=False):
    try:
        messages_json = json.dumps(messages, ensure_ascii=False)
        created_at = messages[0].get('timestamp', datetime.now().isoformat()) if messages else datetime.now().isoformat()
        db.collection(CHAT_COLLECTION).document(chat_id).set({
            'chat_id': chat_id,
            'messages': messages_json,
            'created_at': created_at,
            'deleted': deleted
        })
        logger.info(f"Saved chat session {chat_id} to Firestore (deleted={deleted})")
    except Exception as e:
        logger.error(f"Error saving chat session {chat_id} to Firestore: {str(e)}")

@mock_aws
def load_all_chat_sessions_from_s3():
    sessions = {'active': {}, 'deleted': {}}
    try:
        for prefix, status in [('chat_sessions/', 'active'), ('deleted_chat_sessions/', 'deleted')]:
            response = s3_client.list_objects_v2(Bucket=S3_BUCKET_NAME, Prefix=prefix)
            if 'Contents' not in response:
                continue
            for obj in response['Contents']:
                chat_id = obj['Key'].split('/')[-1].replace('.json', '')
                try:
                    obj_data = s3_client.get_object(Bucket=S3_BUCKET_NAME, Key=obj['Key'])
                    data = json.loads(obj_data['Body'].read().decode('utf-8'))
                    sessions[status][chat_id] = json.loads(data['messages'])
                except Exception as e:
                    logger.error(f"Error loading chat session {chat_id} from S3: {str(e)}")
        logger.info(f"Loaded {len(sessions['active'])} active, {len(sessions['deleted'])} deleted sessions from S3")
        return sessions
    except Exception as e:
        logger.error(f"Error loading chat sessions from S3: {str(e)}")
        return sessions

def load_all_chat_sessions_from_firestore():
    sessions = {'active': {}, 'deleted': {}}
    try:
        docs = db.collection(CHAT_COLLECTION).stream()
        for doc in docs:
            data = doc.to_dict()
            chat_id = data['chat_id']
            status = 'deleted' if data.get('deleted', False) else 'active'
            try:
                messages = json.loads(data['messages'])
                sessions[status][chat_id] = messages
            except json.JSONDecodeError:
                logger.warning(f"Malformed messages JSON for chat_id {chat_id}")
                sessions[status][chat_id] = []
        logger.info(f"Loaded {len(sessions['active'])} active, {len(sessions['deleted'])} deleted sessions from Firestore")
        return sessions
    except Exception as e:
        logger.error(f"Error loading chat sessions from Firestore: {str(e)}")
        return sessions

@mock_aws
def delete_chat_session_from_s3(chat_id):
    try:
        obj = s3_client.get_object(Bucket=S3_BUCKET_NAME, Key=f"chat_sessions/{chat_id}.json")
        data = json.loads(obj['Body'].read().decode('utf-8'))
        save_chat_session_to_s3(chat_id, json.loads(data['messages']), deleted=True)
        s3_client.delete_object(Bucket=S3_BUCKET_NAME, Key=f"chat_sessions/{chat_id}.json")
        logger.info(f"Soft-deleted chat session {chat_id} in S3")
        return True
    except ClientError as e:
        if e.response['Error']['Code'] == 'NoSuchKey':
            logger.info(f"Chat session {chat_id} not found in S3")
            return False
        raise
    except Exception as e:
        logger.error(f"Error soft-deleting chat session {chat_id} in S3: {str(e)}")
        return False

def delete_chat_session_from_firestore(chat_id):
    try:
        doc_ref = db.collection(CHAT_COLLECTION).document(chat_id)
        if doc_ref.get().exists:
            doc_ref.update({'deleted': True})
            logger.info(f"Soft-deleted chat session {chat_id} in Firestore")
            return True
        logger.info(f"Chat session {chat_id} not found in Firestore")
        return False
    except Exception as e:
        logger.error(f"Error soft-deleting chat session {chat_id} in Firestore: {str(e)}")
        return False

@mock_aws
def restore_chat_session_from_s3(chat_id):
    try:
        obj = s3_client.get_object(Bucket=S3_BUCKET_NAME, Key=f"deleted_chat_sessions/{chat_id}.json")
        data = json.loads(obj['Body'].read().decode('utf-8'))
        save_chat_session_to_s3(chat_id, json.loads(data['messages']), deleted=False)
        s3_client.delete_object(Bucket=S3_BUCKET_NAME, Key=f"deleted_chat_sessions/{chat_id}.json")
        logger.info(f"Restored chat session {chat_id} in S3")
        return True
    except ClientError as e:
        if e.response['Error']['Code'] == 'NoSuchKey':
            logger.info(f"Deleted chat session {chat_id} not found in S3")
            return False
        raise
    except Exception as e:
        logger.error(f"Error restoring chat session {chat_id} in S3: {str(e)}")
        return False

def restore_chat_session_from_firestore(chat_id):
    try:
        doc_ref = db.collection(CHAT_COLLECTION).document(chat_id)
        if doc_ref.get().exists:
            doc_ref.update({'deleted': False})
            logger.info(f"Restored chat session {chat_id} in Firestore")
            return True
        logger.info(f"Deleted chat session {chat_id} not found in Firestore")
        return False
    except Exception as e:
        logger.error(f"Error restoring chat session {chat_id} in Firestore: {str(e)}")
        return False

@app.route('/upload_file', methods=['POST'])
def upload_file():
    chat_id = request.form.get('chat_id', str(uuid.uuid4()))
    if 'file' not in request.files:
        logger.warning("No file part in upload request")
        return jsonify({"error": "No file provided."}), 400
    file = request.files['file']
    if file.filename == '':
        logger.warning("No selected file in upload request")
        return jsonify({"error": "No file selected."}), 400
    try:
        sessions = load_all_chat_sessions()
        if chat_id not in sessions['active']:
            sessions['active'][chat_id] = []
        image = Image.open(file)
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='JPEG')
        if STORAGE_BACKEND == 'firestore':
            img_data = img_byte_arr.getvalue()
            db.collection(CHAT_COLLECTION).document(chat_id).update({
                'messages': firestore.ArrayUnion([{
                    'sender': 'user',
                    'message': f"Uploaded image: {file.filename}",
                    'timestamp': datetime.now().isoformat(),
                    'image_data': img_data
                }])
            })
        else:
            s3_client.put_object(
                Bucket=S3_BUCKET_NAME,
                Key=f"images/{chat_id}/{file.filename}",
                Body=img_byte_arr.getvalue()
            )
            sessions['active'][chat_id].append({
                'sender': 'user',
                'message': f"Uploaded image: {file.filename}",
                'timestamp': datetime.now().isoformat()
            })
        response = f"Received your image ({file.filename}). Can you describe what's in it or how it makes you feel?"
        sessions['active'][chat_id].append({
            'sender': 'bot',
            'message': response,
            'timestamp': datetime.now().isoformat()
        })
        save_chat_session(chat_id, sessions['active'][chat_id])
        logger.info(f"Processed image upload for chat_id {chat_id}")
        return jsonify({"response": response, "chat_id": chat_id}), 200
    except Exception as e:
        logger.error(f"Error processing image upload: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/check_s3', methods=['GET'])
def check_s3():
    try:
        response = s3_client.list_objects_v2(Bucket=S3_BUCKET_NAME)
        objects = []
        if 'Contents' in response:
            for obj in response['Contents']:
                obj_data = s3_client.get_object(Bucket=S3_BUCKET_NAME, Key=obj['Key'])
                data = obj_data['Body'].read().decode('utf-8')
                objects.append({"key": obj['Key'], "data": data})
        return jsonify({"status": "success", "objects": objects or [{"message": "No objects found."}]}), 200
    except Exception as e:
        logger.error(f"Error checking S3: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/check_firestore', methods=['GET'])
def check_firestore():
    try:
        docs = db.collection(CHAT_COLLECTION).limit(10).stream()
        objects = [{"chat_id": doc.id, "data": doc.to_dict()} for doc in docs]
        return jsonify({"status": "success", "objects": objects or [{"message": "No chat sessions found."}]}), 200
    except Exception as e:
        logger.error(f"Error checking Firestore: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/test_s3', methods=['GET'])
def test_s3():
    try:
        chat_id = str(uuid.uuid4())
        test_messages = [
            {"sender": "user", "message": "Test message", "timestamp": datetime.now().isoformat()},
            {"sender": "bot", "message": "Test response", "timestamp": datetime.now().isoformat()}
        ]
        save_chat_session_to_s3(chat_id, test_messages)
        return jsonify({"status": "success", "chat_id": chat_id, "message": "Test chat session saved to S3."}), 200
    except Exception as e:
        logger.error(f"Error testing S3: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/test_firestore', methods=['GET'])
def test_firestore():
    try:
        chat_id = str(uuid.uuid4())
        test_messages = [
            {"sender": "user", "message": "Test message", "timestamp": datetime.now().isoformat()},
            {"sender": "bot", "message": "Test response", "timestamp": datetime.now().isoformat()}
        ]
        save_chat_session_to_firestore(chat_id, test_messages)
        return jsonify({"status": "success", "chat_id": chat_id, "message": "Test chat session saved to Firestore."}), 200
    except Exception as e:
        logger.error(f"Error testing Firestore: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "storage_backend": STORAGE_BACKEND}), 200

# Gemini API Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    logger.error("GEMINI_API_KEY environment variable is not set")
    raise ValueError("GEMINI_API_KEY is required")

GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"

SENSITIVE_TOPIC_KEYWORDS = [
    "criticize", "criticism", "discriminate", "discrimination", "racism", "racist",
    "unfair", "wrong", "judged", "hate", "harass", "bullying", "prejudice", "stereotyped",
    "badly treated", "abuse", "accused", "blame", "fault", "problem", "difficult", "mock", "mocked"
]

@app.route('/')
def home():
    logger.info("Rendering welcome.html")
    return render_template("welcome.html")

@app.route('/chatbot_ui')
def chatbot_ui():
    logger.info("Rendering chatbot.html")
    return render_template("chatbot.html")

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        logger.info("POST request to /login, redirecting to chatbot_ui")
        return redirect(url_for('chatbot_ui'))
    logger.info("Rendering login.html")
    return render_template("login.html")

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        logger.info("POST request to /signup, redirecting to chatbot_ui")
        return redirect(url_for('chatbot_ui'))
    logger.info("Rendering signup.html")
    return render_template("signup.html")

@app.route('/privacy')
def privacy():
    logger.info("Rendering privacy.html")
    return render_template("privacy.html")

@app.route('/terms')
def terms():
    logger.info("Rendering terms.html")
    return render_template("terms.html")

@app.route('/get_chat_sessions', methods=['GET'])
def get_chat_sessions():
    try:
        sessions = load_all_chat_sessions()
        active_sessions = []
        deleted_sessions = []
        for chat_id, messages in sessions['active'].items():
            preview = messages[0]['message'][:50] + '...' if messages and len(messages[0]['message']) > 50 else messages[0]['message'] if messages else "New Chat"
            active_sessions.append({"id": chat_id, "preview": preview, "timestamp": messages[0]['timestamp'] if messages else datetime.now().isoformat()})
        for chat_id, messages in sessions['deleted'].items():
            preview = messages[0]['message'][:50] + '...' if messages and len(messages[0]['message']) > 50 else messages[0]['message'] if messages else "Deleted Chat"
            deleted_sessions.append({"id": chat_id, "preview": preview, "timestamp": messages[0]['timestamp'] if messages else datetime.now().isoformat()})
        active_sessions.sort(key=lambda x: x['timestamp'], reverse=True)
        deleted_sessions.sort(key=lambda x: x['timestamp'], reverse=True)
        logger.info(f"Retrieved {len(active_sessions)} active and {len(deleted_sessions)} deleted sessions")
        return jsonify({"active": active_sessions, "deleted": deleted_sessions}), 200
    except Exception as e:
        logger.error(f"Error getting chat sessions: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/get_history', methods=['GET'])
def get_history():
    chat_id = request.args.get('chat_id')
    sessions = load_all_chat_sessions()
    if chat_id and (chat_id in sessions['active'] or chat_id in sessions['deleted']):
        logger.info(f"Retrieved history for chat_id {chat_id}")
        return jsonify(sessions['active'].get(chat_id, sessions['deleted'].get(chat_id, []))), 200
    return jsonify([]), 200

@app.route('/delete_chat_session/<string:chat_id>', methods=['DELETE'])
def delete_chat_session_backend(chat_id):
    if delete_chat_session(chat_id):
        logger.info(f"Chat session {chat_id} soft-deleted")
        return jsonify({"message": f"Chat session {chat_id} moved to deleted."}), 200
    logger.warning(f"Chat session {chat_id} not found")
    return jsonify({"error": "Chat session not found."}), 404

@app.route('/restore_chat_session/<string:chat_id>', methods=['POST'])
def restore_chat_session_backend(chat_id):
    if restore_chat_session(chat_id):
        logger.info(f"Chat session {chat_id} restored")
        return jsonify({"status": "success", "message": f"Chat session {chat_id} restored."}), 200
    logger.warning(f"Deleted chat session {chat_id} not found")
    return jsonify({"error": "Deleted chat session not found."}), 404

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_input = data.get('message', '').strip()
    current_chat_id = data.get('chat_id')
    sessions = load_all_chat_sessions()
    if current_chat_id == 'new' or not current_chat_id:
        current_chat_id = str(uuid.uuid4())
        sessions['active'][current_chat_id] = []
    if current_chat_id not in sessions['active']:
        sessions['active'][current_chat_id] = []

    if not user_input:
        return jsonify({"response": "Please type something to chat with EmpathAI.", "analysis": {}, "chat_id": current_chat_id}), 200

    timestamp = datetime.now().isoformat()
    sessions['active'][current_chat_id].append({"sender": "user", "message": user_input, "timestamp": timestamp})
    final_bot_reply = "I'm having a little trouble responding right now. Please try again."
    emotion_data = {}
    try:
        emotion_data = analyze_emotion(user_input)
        emotion = emotion_data.get('intent_emotion', 'unknown')
        mood = emotion_data.get('mood', 'neutral')
        advice = emotion_data.get('advice', "I'm here to support you.")
        logger.info(f"Emotion analysis: intent_emotion={emotion}, mood={mood}")
    except Exception as e:
        logger.error(f"Emotion analysis error: {str(e)}")
        emotion_data = {'intent_emotion': 'unknown', 'mood': 'neutral', 'advice': "I'm here to support you."}
        emotion = 'unknown'
        mood = 'neutral'
        advice = "I'm here to support you."

    conversation_context = [{"role": "user" if msg['sender'] == 'user' else "model", "parts": [{"text": msg['message']}]} for msg in sessions['active'][current_chat_id][-10:]]
    initial_user_problem = next((msg['message'] for msg in sessions['active'][current_chat_id] if msg['sender'] == 'user' and any(kw in msg['message'].lower() for kw in ["depression", "trauma", "difficult time", "struggling"])), "")
    
    base_prompt = (
        f"You are EmpathAI, a highly empathetic, supportive, and insightful AI companion. "
        f"Your goal is to help users process their emotions and find constructive ways forward, always maintaining a positive and supportive tone without negative sentences. "
        f"The user's current input is: '{user_input}'. "
        f"Based on our analysis, their core emotion is '{emotion}' and their current mood is '{mood}'. "
        f"If relevant, consider their earlier problem: '{initial_user_problem}'."
    )

    is_sensitive_topic = any(keyword in user_input.lower() for keyword in SENSITIVE_TOPIC_KEYWORDS)
    if is_sensitive_topic:
        dynamic_prompt = (
            f"Acknowledge the difficulty they're expressing and gently pivot to how *they* feel or what *they* want to do. "
            f"Keep your response concise (1-2 sentences)."
        )
    elif mood == 'negative' or len(user_input.split()) < 5:
        dynamic_prompt = (
            f"Reflect on their words or ask an open-ended, clarifying question to encourage elaboration. "
            f"Example: 'What makes you feel [emotion]?'"
        )
    else:
        dynamic_prompt = (
            f"Respond empathetically with a gentle supportive statement or encouragement. "
            f"Incorporate advice: '{advice}' if relevant."
        )

    payload = {
        "contents": conversation_context + [{"role": "user", "parts": [{"text": base_prompt + dynamic_prompt}]}],
        "generationConfig": {"maxOutputTokens": 80, "temperature": 0.8, "topP": 0.95}
    }

    try:
        headers = {'Content-Type': 'application/json'}
        response = requests.post(f"{GEMINI_API_URL}?key={GEMINI_API_KEY}", headers=headers, json=payload)
        response.raise_for_status()
        gemini_result = response.json()
        if gemini_result.get('candidates'):
            final_bot_reply = gemini_result['candidates'][0]['content']['parts'][0]['text'].strip()
        else:
            final_bot_reply = "I'm here to listen, but I couldn't generate a response right now."
    except Exception as e:
        logger.error(f"Gemini API error: {str(e)}")
        final_bot_reply = "I'm having trouble connecting right now, but I'm here for you."

    sessions['active'][current_chat_id].append({"sender": "bot", "message": final_bot_reply, "timestamp": datetime.now().isoformat()})
    save_chat_session(current_chat_id, sessions['active'][current_chat_id])
    return jsonify({"response": final_bot_reply, "analysis": emotion_data, "chat_id": current_chat_id}), 200

@app.route('/summarize_chat', methods=['POST'])
def summarize_chat():
    data = request.json
    chat_id = data.get('chat_id')
    if not chat_id:
        logger.warning("Chat ID missing for summarization")
        return jsonify({"error": "Chat ID is required for summarization."}), 400

    sessions = load_all_chat_sessions()
    if chat_id not in sessions['active']:
        logger.warning(f"Chat session {chat_id} not found for summarization")
        return jsonify({"error": "Chat session not found."}), 404

    try:
        chat_history = sessions['active'][chat_id]
        conversation_text = [f"{msg['sender'].capitalize()}: {msg['message']}" for msg in chat_history]
        if not conversation_text:
            return jsonify({"summary": "This chat session is empty."}), 200

        summary_prompt = (
            f"Summarize the following conversation concisely (2-4 sentences):\n\n{'\n'.join(conversation_text)}"
        )

        payload = {
            "contents": [{"role": "user", "parts": [{"text": summary_prompt}]}],
            "generationConfig": {"maxOutputTokens": 150, "temperature": 0.5}
        }

        response = requests.post(f"{GEMINI_API_URL}?key={GEMINI_API_KEY}", headers={'Content-Type': 'application/json'}, json=payload)
        response.raise_for_status()
        gemini_result = response.json()
        summary_text = gemini_result['candidates'][0]['content']['parts'][0]['text'].strip() if gemini_result.get('candidates') else "Could not generate summary."
        return jsonify({"summary": summary_text}), 200
    except Exception as e:
        logger.error(f"Error summarizing chat: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/generate_insightful_question', methods=['POST'])
def generate_insightful_question():
    data = request.json
    chat_id = data.get('chat_id')
    if not chat_id:
        return jsonify({"error": "Chat ID is required."}), 400

    sessions = load_all_chat_sessions()
    if chat_id not in sessions['active']:
        return jsonify({"error": "Chat session not found."}), 404

    try:
        chat_history = sessions['active'][chat_id]
        conversation_context = [{"role": "user" if msg['sender'] == 'user' else "model", "parts": [{"text": msg['message']}]} for msg in chat_history[-6:]]
        if not conversation_context:
            return jsonify({"question": "Let's start chatting to generate insightful questions!"}), 200

        question_prompt = (
            f"Based on the conversation, generate one empathetic, open-ended question to encourage deeper reflection:\n\nConversation context:"
        )

        payload = {
            "contents": conversation_context + [{"role": "user", "parts": [{"text": question_prompt}]}],
            "generationConfig": {"maxOutputTokens": 50, "temperature": 0.7}
        }

        response = requests.post(f"{GEMINI_API_URL}?key={GEMINI_API_KEY}", headers={'Content-Type': 'application/json'}, json=payload)
        response.raise_for_status()
        gemini_result = response.json()
        question_text = gemini_result['candidates'][0]['content']['parts'][0]['text'].strip() if gemini_result.get('candidates') else "Could not generate a question."
        return jsonify({"question": question_text}), 200
    except Exception as e:
        logger.error(f"Error generating question: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/suggest_coping_mechanism', methods=['POST'])
def suggest_coping_mechanism():
    data = request.json
    chat_id = data.get('chat_id')
    if not chat_id:
        return jsonify({"error": "Chat ID is required."}), 400

    sessions = load_all_chat_sessions()
    if chat_id not in sessions['active']:
        return jsonify({"error": "Chat session not found."}), 404

    try:
        chat_history = sessions['active'][chat_id]
        last_user_message = next((msg['message'] for msg in reversed(chat_history) if msg['sender'] == 'user'), "")
        if not last_user_message:
            return jsonify({"suggestion": "Please send a message to get a coping suggestion."}), 200

        emotion_data = analyze_emotion(last_user_message)
        mood = emotion_data.get('mood', 'neutral')
        intent_emotion = emotion_data.get('intent_emotion', 'general')
        coping_prompt = (
            f"User message: '{last_user_message}'. Emotion: '{intent_emotion}', Mood: '{mood}'. "
            f"Suggest one simple coping mechanism (1-2 sentences)."
        )

        payload = {
            "contents": [{"role": "user", "parts": [{"text": coping_prompt}]}],
            "generationConfig": {"maxOutputTokens": 60, "temperature": 0.8}
        }

        response = requests.post(f"{GEMINI_API_URL}?key={GEMINI_API_KEY}", headers={'Content-Type': 'application/json'}, json=payload)
        response.raise_for_status()
        gemini_result = response.json()
        suggestion_text = gemini_result['candidates'][0]['content']['parts'][0]['text'].strip() if gemini_result.get('candidates') else "Could not suggest a coping mechanism."
        return jsonify({"suggestion": suggestion_text}), 200
    except Exception as e:
        logger.error(f"Error suggesting coping mechanism: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/reframe_positively', methods=['POST'])
def reframe_positively():
    data = request.json
    chat_id = data.get('chat_id')
    if not chat_id:
        return jsonify({"error": "Chat ID is required."}), 400

    sessions = load_all_chat_sessions()
    if chat_id not in sessions['active']:
        return jsonify({"error": "Chat session not found."}), 404

    try:
        chat_history = sessions['active'][chat_id]
        last_user_message = next((msg['message'] for msg in reversed(chat_history) if msg['sender'] == 'user'), "")
        if not last_user_message:
            return jsonify({"reframe": "Please send a message to reframe."}), 200

        reframe_prompt = (
            f"User said: '{last_user_message}'. Reframe this positively or growth-oriented (1-2 sentences)."
        )

        payload = {
            "contents": [{"role": "user", "parts": [{"text": reframe_prompt}]}],
            "generationConfig": {"maxOutputTokens": 70, "temperature": 0.9}
        }

        response = requests.post(f"{GEMINI_API_URL}?key={GEMINI_API_KEY}", headers={'Content-Type': 'application/json'}, json=payload)
        response.raise_for_status()
        gemini_result = response.json()
        reframe_text = gemini_result['candidates'][0]['content']['parts'][0]['text'].strip() if gemini_result.get('candidates') else "Could not reframe the statement."
        return jsonify({"reframe": reframe_text}), 200
    except Exception as e:
        logger.error(f"Error reframing positively: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)