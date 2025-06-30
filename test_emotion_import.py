import os
import json
import uuid 
from flask import Flask, request, jsonify, render_template
from emotion_analysis import analyze_emotion 
import requests 
from datetime import datetime 
import boto3 # Ensure boto3 is explicitly imported here
from botocore.exceptions import ClientError

# --- Set a custom Transformers cache folder to avoid C: drive usage ---
# IMPORTANT: Ensure this path (D:/lightweight_hf_cache) exists and is writable
os.environ['TRANSFORMERS_CACHE'] = 'D:/lightweight_hf_cache'

app = Flask(__name__, template_folder='templates', static_folder='static')

# --- DynamoDB Configuration ---
# For production, these should be environment variables, not hardcoded.
DYNAMODB_ENDPOINT_URL = 'http://127.0.0.1:8001' 
DYNAMODB_REGION = 'us-east-1' # Region doesn't matter much for local, but needed for boto3
DYNAMODB_TABLE_NAME = 'EmpathAIChatSessions' # Name of our DynamoDB table

# Initialize DynamoDB client (will connect to local instance)
dynamodb = None
table = None # Initialize globally as None

try:
    dynamodb = boto3.resource(
        'dynamodb',
        endpoint_url=DYNAMODB_ENDPOINT_URL,
        region_name=DYNAMODB_REGION,
        # dummy credentials for local DynamoDB - DO NOT use in production AWS
        aws_access_key_id='dummy_access_key',
        aws_secret_access_key='dummy_secret_key'
    )
    # The table resource will be loaded/created later
    print(f"✅ Connected to DynamoDB at {DYNAMODB_ENDPOINT_URL}")
except Exception as e:
    print(f"❌ Failed to connect to DynamoDB: {e}")
    # dynamodb and table remain None if connection fails

# Function to create DynamoDB table if it doesn't exist
def create_dynamodb_table():
    global table # Declare 'table' as global because we might re-assign it here
    if dynamodb is None:
        print("Skipping table creation: DynamoDB client not initialized.")
        return

    try:
        # Check if table exists by listing tables
        existing_tables = dynamodb.meta.client.list_tables()['TableNames']
        if DYNAMODB_TABLE_NAME in existing_tables:
            print(f"Table '{DYNAMODB_TABLE_NAME}' already exists.")
            table = dynamodb.Table(DYNAMODB_TABLE_NAME) # Ensure global 'table' variable is set
            return # Table exists, no need to create
        
        print(f"Table '{DYNAMODB_TABLE_NAME}' not found. Creating...")
        table = dynamodb.create_table( # This assigns to the GLOBAL table due to 'global' keyword
            TableName=DYNAMODB_TABLE_NAME,
            KeySchema=[
                {
                    'AttributeName': 'chat_id',
                    'KeyType': 'HASH'  # Partition key
                }
            ],
            AttributeDefinitions=[
                {
                    'AttributeName': 'chat_id',
                    'AttributeType': 'S' # String
                },
                {
                    'AttributeName': 'created_at', # Add for sorting purposes in get_chat_sessions
                    'AttributeType': 'S' 
                }
            ],
            # Add Global Secondary Index for 'created_at' to allow efficient sorting
            GlobalSecondaryIndexes=[
                {
                    'IndexName': 'CreatedAtGSI',
                    'KeySchema': [
                        {
                            'AttributeName': 'created_at',
                            'KeyType': 'HASH'
                        }
                    ],
                    'Projection': {
                        'ProjectionType': 'ALL' # Or KEYS_ONLY for smaller index
                    },
                    'ProvisionedThroughput': {
                        'ReadCapacityUnits': 1,
                        'WriteCapacityUnits': 1
                    }
                }
            ],
            ProvisionedThroughput={
                'ReadCapacityUnits': 1,
                'WriteCapacityUnits': 1
            }
        )
        table.wait_until_exists() # Wait until the table is created
        print(f"✅ Table '{DYNAMODB_TABLE_NAME}' created successfully.")
    except Exception as create_e:
        print(f"❌ Error creating table '{DYNAMODB_TABLE_NAME}': {create_e}")
        # If table creation fails, ensure global table is set to None to prevent further errors
        table = None 

# Call to create table on app startup
if dynamodb: # Only attempt to create table if dynamodb client was successfully initialized
    create_dynamodb_table()


# Global dictionary to hold all chat sessions (kept for in-memory caching during a run, but primarily sourced from DB)
_all_chat_sessions = {} # This will now act as a cache, not the primary storage


# Function to load all chat history from DynamoDB
def load_all_chat_sessions_from_db():
    global _all_chat_sessions
    _all_chat_sessions = {} # Clear existing cache
    if table is None:
        print("Cannot load from DB: DynamoDB table not available.")
        return

    try:
        # For full initial load, scan is still appropriate.
        # Pagination might be needed for very large datasets.
        response = table.scan() 
        for item in response.get('Items', []):
            chat_id = item.get('chat_id')
            # Messages stored as JSON string in DB, parse back to list of dicts
            messages_json = item.get('messages', '[]')
            try:
                messages = json.loads(messages_json)
            except json.JSONDecodeError:
                messages = [] # Fallback for malformed JSON
                print(f"Warning: Malformed messages JSON for chat_id {chat_id}")
            _all_chat_sessions[chat_id] = messages
        print(f"✅ Loaded {len(_all_chat_sessions)} chat sessions from DynamoDB.")
    except Exception as e:
        print(f"❌ Error loading all chat sessions from DynamoDB: {e}")

# Call to load all sessions from DB on app startup
# This is mainly for initial cache population; individual sessions are fetched on demand.
if table: 
    load_all_chat_sessions_from_db()


# Function to save a specific chat session to DynamoDB
def save_chat_session_to_db(chat_id, messages):
    if table is None:
        print("Cannot save to DB: DynamoDB table not available.")
        return

    try:
        # Convert messages list of dicts to JSON string for storage
        messages_json = json.dumps(messages, ensure_ascii=False)
        
        # Determine the creation timestamp for sorting later in get_chat_sessions
        # If the session is new and has no messages, or first message has no timestamp, use now
        # Otherwise, preserve the original created_at if it exists in the item
        current_item = None
        try:
            # Try to get existing item to preserve 'created_at'
            get_response = table.get_item(Key={'chat_id': chat_id})
            current_item = get_response.get('Item')
        except Exception as e:
            print(f"Warning: Could not fetch existing item for {chat_id} to preserve created_at: {e}")

        if current_item and 'created_at' in current_item:
            created_at = current_item['created_at']
        else:
            created_at = messages[0].get('timestamp', datetime.now().isoformat()) if messages else datetime.now().isoformat()

        table.put_item(
            Item={
                'chat_id': chat_id,
                'messages': messages_json,
                'created_at': created_at # Add a timestamp for sorting
            }
        )
        # print(f"✅ Chat session {chat_id} saved to DynamoDB.") # Can be noisy, uncomment for debugging
    except Exception as e:
        print(f"❌ Error saving chat session {chat_id} to DynamoDB: {e}")

# Function to delete a specific chat session from DynamoDB
def delete_chat_session_from_db(chat_id):
    if table is None:
        print("Cannot delete from DB: DynamoDB table not available.")
        return False

    try:
        table.delete_item(
            Key={'chat_id': chat_id}
        )
        print(f"✅ Chat session {chat_id} deleted from DynamoDB.")
        return True
    except ClientError as e:
        if e.response['Error']['Code'] == 'ResourceNotFoundException':
            print(f"Chat session {chat_id} not found in DB for deletion.")
            return False
        else:
            print(f"❌ Error deleting chat session {chat_id} from DynamoDB: {e}")
            return False
    except Exception as e:
        print(f"❌ Unexpected error deleting chat session {chat_id} from DynamoDB: {e}")
        return False

# --- Gemini API Configuration ---
# IMPORTANT: This API key should be loaded from environment variables in a real application
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "") 
if not GEMINI_API_KEY:
    print("⚠️ WARNING: GEMINI_API_KEY is not set. Chatbot will not be able to use Gemini API.")

GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"

# Keywords to strongly indicate sensitive/negative topics that require direct acknowledgment
SENSITIVE_TOPIC_KEYWORDS = [
    "criticize", "criticism", "discriminate", "discrimination", "racism", "racist",
    "unfair", "wrong", "hate", "harass", "bullying", "prejudice", "stereotyped",
    "abuse", "mistreated", "mock", "mocked" 
]

# --- Flask Routes ---

@app.route('/')
def home():
    return render_template("welcome.html") 

# CHANGED: Route to serve the new chatbot UI with sidebar
@app.route('/chatbot_ui')
def chatbot_ui():
    return render_template("chatbot_ui_with_sidebar.html") 

# NEW: Route for the login page
@app.route('/login')
def login():
    return render_template("login.html")

# MODIFIED: Route to get a list of all available chat session IDs from DB
@app.route('/get_chat_sessions', methods=['GET'])
def get_chat_sessions():
    session_list = []
    if table is None:
        return jsonify(session_list) # Return empty if DB not available

    try:
        # Use the GSI to query for chat_id and created_at, sorted by created_at in descending order
        # This is more efficient than a full table scan and then sorting in Python for large tables.
        response = dynamodb.meta.client.query(
            TableName=DYNAMODB_TABLE_NAME,
            IndexName='CreatedAtGSI',
            KeyConditionExpression=boto3.dynamodb.conditions.Key('created_at').exists(), # Query for any item with a created_at
            ProjectionExpression="chat_id, messages, created_at",
            ScanIndexForward=False # Sort in descending order (most recent first)
        )
        
        # Note: A query on a GSI where the HASH key is just 'created_at' and it's a string,
        # means it queries *all* items by 'created_at'. If you had a different primary key,
        # you'd need to adjust the query or stick with scan if that's what's intended.
        # The `KeyConditionExpression=boto3.dynamodb.conditions.Key('created_at').exists()` is a common
        # workaround for GSI where you want to retrieve all items sorted by the GSI hash key.

        items = response.get('Items', [])

        for item in items:
            chat_id = item.get('chat_id')
            # DynamoDB returns numbers as Decimal by default, ensure strings where expected
            messages_json = item.get('messages', '[]')
            created_at = item.get('created_at', datetime.min.isoformat())

            messages = []
            try:
                messages = json.loads(messages_json)
            except json.JSONDecodeError:
                pass # Continue with empty messages if malformed

            first_message_preview = ""
            if messages:
                for msg in messages:
                    if msg.get('sender') == 'user':
                        first_message_preview = msg.get('message', '')[:50] + '...' if len(msg.get('message', '')) > 50 else msg.get('message', '')
                        break
                if not first_message_preview:
                    # If no user message, use the first bot message as preview
                    if messages: # Double check messages exist after populating
                        first_message_preview = messages[0].get('message', '')[:50] + '...' if messages[0].get('message') and len(messages[0].get('message', '')) > 50 else messages[0].get('message', '')
            
            # If still no preview (e.g., chat is brand new and only contains auto-greeting), use a default
            if not first_message_preview and chat_id: # Only for existing chats, not the 'new' placeholder
                first_message_preview = "Empty Chat" 

            display_name = first_message_preview if first_message_preview else "New Chat" # Fallback display name
            
            session_list.append({
                "id": chat_id,
                "preview": display_name,
                "timestamp": created_at # Use the timestamp from DB
            })
    except Exception as e:
        print(f"❌ Error getting chat sessions from DynamoDB: {e}")
    return jsonify(session_list)


# MODIFIED: Route to get initial chat history for a specific session from DB
@app.route('/get_history', methods=['GET'])
def get_history():
    chat_id = request.args.get('chat_id') 
    
    if not table:
        return jsonify([]) # Return empty if DB not available

    if chat_id:
        return get_history_by_id(chat_id).get_json() # Use the helper function directly
    else:
        # If no chat_id is provided, try to find the most recent session from DB
        try:
            # Query the GSI to get the most recent chat_id
            response = dynamodb.meta.client.query(
                TableName=DYNAMODB_TABLE_NAME,
                IndexName='CreatedAtGSI',
                KeyConditionExpression=boto3.dynamodb.conditions.Key('created_at').exists(),
                ProjectionExpression="chat_id", # Only need chat_id for this purpose
                ScanIndexForward=False, # Most recent first
                Limit=1 # Only need one item
            )
            items = response.get('Items', [])
            if items:
                latest_chat_id = items[0].get('chat_id')
                return get_history_by_id(latest_chat_id).get_json()
            return jsonify([]) # No chats found
        except Exception as e:
            print(f"❌ Error finding latest chat for get_history (no chat_id): {e}")
            return jsonify([])

# Helper function for get_history to fetch specific chat content
def get_history_by_id(chat_id):
    if not table: return jsonify([])
    try:
        response = table.get_item(Key={'chat_id': chat_id})
        item = response.get('Item')
        if item:
            messages_json = item.get('messages', '[]')
            try:
                history = json.loads(messages_json)
            except json.JSONDecodeError:
                history = []
            _all_chat_sessions[chat_id] = history
            return jsonify(history)
        return jsonify([])
    except Exception as e:
        print(f"❌ Error in get_history_by_id for {chat_id}: {e}")
        return jsonify([])


# MODIFIED: API endpoint for chatbot interactions (main chat logic)
@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_input = data.get('message', '').strip()
    # Now expecting chat_id from the frontend
    current_chat_id = data.get('chat_id') 
    
    # If no chat_id is provided or it's explicitly 'new', generate a new one
    is_new_chat_request = (current_chat_id == 'new' or not current_chat_id)
    if is_new_chat_request:
        current_chat_id = str(uuid.uuid4())
        # Initialize an empty list for this new session
        _all_chat_sessions[current_chat_id] = [] 
        # Add the initial bot message only if it's a brand new session and no user input yet.
        # This is the "Hello there!" message.
        if not user_input and not _all_chat_sessions[current_chat_id]:
             _all_chat_sessions[current_chat_id].append({
                 "sender": "bot",
                 "message": "Hello there! I'm EmpathAI, your emotional companion. How are you feeling today?",
                 "timestamp": datetime.now().isoformat()
             })
        # Save the newly initialized chat session to DB immediately
        save_chat_session_to_db(current_chat_id, _all_chat_sessions[current_chat_id])


    # Ensure the chat session exists in the in-memory cache, loading from DB if not present
    if current_chat_id not in _all_chat_sessions:
        if table:
            try:
                response = table.get_item(Key={'chat_id': current_chat_id})
                item = response.get('Item')
                if item:
                    _all_chat_sessions[current_chat_id] = json.loads(item.get('messages', '[]'))
                else:
                    _all_chat_sessions[current_chat_id] = [] # Not in DB, initialize empty
            except Exception as e:
                print(f"Error fetching chat {current_chat_id} from DB for cache: {e}")
                _all_chat_sessions[current_chat_id] = []
        else:
            _all_chat_sessions[current_chat_id] = [] # Fallback if DB not available


    # Do not process or save empty user messages (except for the initial bot message if chat is new and empty)
    if not user_input: # If user input is empty, and it's not the initial bot message auto-add case, return.
        # This handles cases where user clicks send on empty input in an existing chat.
        return jsonify({"response": "Please type something to chat with EmpathAI.", "analysis": {}, "chat_id": current_chat_id}), 200
    
    # If it's a new chat (just auto-added initial bot message) and the user sends the first message, 
    # remove the auto-added bot greeting to start with user's input as the first entry.
    # This check now also ensures the chat isn't just an "empty chat" that needs a first user message
    if user_input and len(_all_chat_sessions[current_chat_id]) == 1 and \
       _all_chat_sessions[current_chat_id][0]["sender"] == "bot" and \
       _all_chat_sessions[current_chat_id][0]["message"] == "Hello there! I'm EmpathAI, your emotional companion. How are you feeling today?":
        _all_chat_sessions[current_chat_id].pop(0) # Remove the initial bot greeting


    # Add user message to current chat history with timestamp
    timestamp = datetime.now().isoformat()
    _all_chat_sessions[current_chat_id].append({"sender": "user", "message": user_input, "timestamp": timestamp})

    try:
        # Step 1: Emotion Analysis using emotion_analysis.py (local processing)
        emotion_data = analyze_emotion(user_input)
        emotion = emotion_data.get('intent_emotion', 'unknown')
        mood = emotion_data.get('mood', 'neutral') 
        advice_from_analysis = emotion_data.get('advice', "I'm here to support you.") 

        general_reply_text = ""
        try:
            # Prepare conversation context for Gemini, ensuring proper roles (user/model)
            conversation_context = []
            initial_user_problem = ""
            for msg in _all_chat_sessions[current_chat_id]:
                if msg['sender'] == 'user':
                    # Heuristic to identify initial user problem for long-term memory
                    if "depression" in msg['message'].lower() or "trauma" in msg['message'].lower() or "difficult time" in msg['message'].lower() or "struggling" in msg['message'].lower():
                        if not initial_user_problem: # Capture the first such statement
                            initial_user_problem = msg['message'] 
                    conversation_context.append({"role": "user", "parts": [{"text": msg['message']}]})
                elif msg['sender'] == 'bot':
                    conversation_context.append({"role": "model", "parts": [{"text": msg['message']}]})
            
            # Limit conversation context to a reasonable number of turns for Gemini's token limit and focus
            # The current user_input is implicitly added by the full_gemini_prompt structure
            # Ensure the last user input is always part of the context sent to Gemini for its immediate response
            # Note: The `full_gemini_prompt` *itself* contains `user_input`.
            # We want to send previous turns as `contents` for conversational memory.
            # `conversation_context_for_gemini` will be the history *excluding* the current user input,
            # which is then added as a final part of the `contents` list along with the system prompt.
            conversation_context_for_gemini = conversation_context[:-1] if conversation_context and conversation_context[-1].get('role') == 'user' else conversation_context
            
            # --- Advanced Prompt Construction for Gemini ---
            base_prompt = (
                f"You are EmpathAI, a highly empathetic, supportive, and insightful AI companion. "
                f"Your goal is to help users process their emotions and find constructive ways forward, always maintaining a positive and supportive tone without negative sentences. "
                f"You remember the user's journey. If they mentioned an ongoing problem like depression or trauma earlier (e.g., '{initial_user_problem}' if present), keep that in mind. "
                f"The user's current input is: '{user_input}'. "
                f"Based on our analysis, their core emotion is '{emotion}' and their current mood is '{mood}'. "
            )

            # Conditional Prompting based on deeper analysis and intent
            is_sensitive_topic = any(keyword in user_input.lower() for keyword in SENSITIVE_TOPIC_KEYWORDS)

            if is_sensitive_topic: 
                dynamic_prompt = (
                    f"Acknowledge the unfairness or difficulty they're expressing and unequivocally condemn any form of discrimination or unfair criticism. "
                    f"Then, gently pivot to how *they* feel, what *they* want to do, or ask a clarifying, reflective question to encourage elaboration and self-reflection. "
                    f"Keep your response concise (1-2 sentences). "
                    f"Example: 'No one should be criticized for their race. That sounds incredibly difficult. How are you coping with that?', 'It sounds incredibly unfair. What are your feelings about it?'"
                )
            elif mood == 'negative' or len(user_input.split()) < 5: 
                dynamic_prompt = (
                    f"Respond by reflecting on their words, or by asking an open-ended, clarifying question based on what they said. "
                    f"Encourage them to elaborate on their feelings or situation. Avoid direct advice unless a clear path is requested. "
                    f"Example: 'You mentioned [word/phrase], could you tell me more about that?', 'What makes you feel [emotion]?','It sounds like you're feeling [emotion], can you elaborate?'"
                )
            else:
                dynamic_prompt = (
                    f"Respond empathetically and concisely. You can offer a gentle supportive statement or encouragement. "
                    f"If a problem was stated at the beginning of the conversation ('{initial_user_problem}' if present), subtly tie your response back to it if relevant. "
                    f"A relevant empathetic thought or initial advice (from our analysis) for this emotion is: '{advice_from_analysis}'. "
                    f"Prioritize natural flow and brevity. "
                    f"Example: 'That's a valid point. What are your thoughts on that?' or 'I understand. What do you feel is the next step for you?'"
                )
            
            # Combine the base and dynamic parts of the prompt
            full_gemini_prompt = base_prompt + dynamic_prompt + (
                " Ensure your response is 1-3 sentences and maintains a supportive, positive tone without any negative phrasing."
            )

            payload = {
                # The 'contents' should be the actual conversation turns, followed by the specific instruction/prompt.
                # Gemini expects alternating roles, starting with 'user'.
                "contents": conversation_context_for_gemini + [{"role": "user", "parts": [{"text": user_input}]}], # Add current user message
                "system_instruction": {"parts": [{"text": full_gemini_prompt}]}, # System instruction for persona/goals
                "generationConfig": {
                    "maxOutputTokens": 80, 
                    "temperature": 0.8,     
                    "topP": 0.95,           
                    "topK": 40              
                }
            }
            
            headers = {'Content-Type': 'application/json'}
            api_url_with_key = f"{GEMINI_API_URL}?key={GEMINI_API_KEY}" 

            response = requests.post(api_url_with_key, headers=headers, json=payload)
            response.raise_for_status() 

            gemini_result = response.json()

            if gemini_result and gemini_result.get('candidates') and \
               gemini_result['candidates'][0].get('content') and \
               gemini_result['candidates'][0]['content'].get('parts'):
                general_reply_text = gemini_result['candidates'][0]['content']['parts'][0]['text'].strip()
            else:
                general_reply_text = "I couldn't generate a detailed response right now, but I'm here to listen."

        except requests.exceptions.RequestException as req_err:
            print(f"❌ Gemini API Request Error: {req_err}")
            general_reply_text = "I'm having trouble connecting to my main brain right now. Please try again in a moment."
        except Exception as api_err:
            print(f"❌ Error processing Gemini API response: {api_err}")
            general_reply_text = "My main brain encountered an issue. I'll still listen closely to your emotions."
        
        final_bot_reply = general_reply_text
        
        _all_chat_sessions[current_chat_id].append({"sender": "bot", "message": final_bot_reply, "timestamp": datetime.now().isoformat()})
        
        # MODIFIED: Save updated session to DynamoDB
        save_chat_session_to_db(current_chat_id, _all_chat_sessions[current_chat_id])

        return jsonify({
            "response": final_bot_reply,
            "analysis": emotion_data,
            "chat_id": current_chat_id # Return the chat_id back to frontend
        })

    except Exception as e:
        print(f"❌ Global chat processing error: {e}")
        fallback_reply = "😐 Sorry, something went wrong with my emotional analysis or reply generation. Please try again?"
        _all_chat_sessions[current_chat_id].append({"sender": "bot", "message": fallback_reply, "timestamp": datetime.now().isoformat()})
        save_chat_session_to_db(current_chat_id, _all_chat_sessions[current_chat_id]) # Save fallback to DB
        return jsonify({
            "response": fallback_reply,
            "error": str(e),
            "chat_id": current_chat_id # Return the chat_id even on error
        }), 500

# NEW ROUTE: To delete a chat session (now deletes from DynamoDB)
@app.route('/delete_chat_session/<string:chat_id>', methods=['DELETE'])
def delete_chat_session_backend(chat_id):
    global _all_chat_sessions
    if delete_chat_session_from_db(chat_id): # Call the new DB deletion function
        if chat_id in _all_chat_sessions:
            del _all_chat_sessions[chat_id] # Remove from in-memory cache
        return jsonify({"message": f"Chat session {chat_id} deleted successfully."}), 200
    else:
        return jsonify({"error": "Chat session not found or failed to delete."}), 404

# NEW LLM-POWERED FEATURE: Summarize Chat Session
@app.route('/summarize_chat', methods=['POST'])
def summarize_chat():
    data = request.json
    chat_id = data.get('chat_id')

    if not chat_id:
        return jsonify({"error": "Chat ID is required for summarization."}), 400
    if table is None:
        return jsonify({"error": "Database not available for summarization."}), 500

    try:
        response = table.get_item(Key={'chat_id': chat_id})
        item = response.get('Item')
        
        if not item:
            return jsonify({"error": "Chat session not found."}), 404

        messages_json = item.get('messages', '[]')
        chat_history = json.loads(messages_json)

        # Filter out only user and bot messages for the summary context
        conversation_text = []
        for msg in chat_history:
            if msg.get('sender') == 'user':
                conversation_text.append({"role": "user", "parts": [{"text": msg.get('message', '')}]})
            elif msg.get('sender') == 'bot':
                conversation_text.append({"role": "model", "parts": [{"text": msg.get('message', '')}]})
        
        if not conversation_text:
            return jsonify({"summary": "This chat session is empty, so there's nothing to summarize."}), 200

        # Construct the prompt for Gemini
        summary_prompt = (
            "Summarize the following conversation concisely, focusing on the main topics, key emotions, "
            "and any resolutions or ongoing issues. Keep the summary to 2-4 sentences."
        )

        payload = {
            "contents": conversation_text, # Send full conversation as contents
            "system_instruction": {"parts": [{"text": summary_prompt}]}, # System instruction for summarization
            "generationConfig": {
                "maxOutputTokens": 150,
                "temperature": 0.5,
                "topP": 0.9,
                "topK": 20
            }
        }
        
        headers = {'Content-Type': 'application/json'}
        api_url_with_key = f"{GEMINI_API_URL}?key={GEMINI_API_KEY}" 

        gemini_response = requests.post(api_url_with_key, headers=headers, json=payload)
        gemini_response.raise_for_status()

        gemini_result = gemini_response.json()
        summary_text = "Could not generate summary."
        if gemini_result and gemini_result.get('candidates') and \
           gemini_result['candidates'][0].get('content') and \
           gemini_result['candidates'][0]['content'].get('parts'):
            summary_text = gemini_result['candidates'][0]['content']['parts'][0]['text'].strip()

        return jsonify({"summary": summary_text}), 200

    except requests.exceptions.RequestException as req_err:
        print(f"❌ Gemini API Request Error for summarization: {req_err}")
        return jsonify({"error": "Failed to connect to AI for summary. Please try again later."}), 500
    except Exception as e:
        print(f"❌ Error during summarization: {e}")
        return jsonify({"error": f"An unexpected error occurred during summarization: {e}"}), 500


# NEW LLM-POWERED FEATURE: Generate Insightful Question
@app.route('/generate_insightful_question', methods=['POST'])
def generate_insightful_question():
    data = request.json
    chat_id = data.get('chat_id')

    if not chat_id:
        return jsonify({"error": "Chat ID is required to generate a question."}), 400
    if table is None:
        return jsonify({"error": "Database not available to generate a question."}), 500

    try:
        response = table.get_item(Key={'chat_id': chat_id})
        item = response.get('Item')
        
        if not item:
            return jsonify({"error": "Chat session not found."}), 404

        messages_json = item.get('messages', '[]')
        chat_history = json.loads(messages_json)

        # Extract the last few turns for context for Gemini
        conversation_context = []
        for msg in chat_history[-6:]: # Consider last 6 messages for context
            if msg.get('sender') == 'user':
                conversation_context.append({"role": "user", "parts": [{"text": msg.get('message', '')}]})
            elif msg.get('sender') == 'bot':
                conversation_context.append({"role": "model", "parts": [{"text": msg.get('message', '')}]})
        
        if not conversation_context:
            return jsonify({"question": "Let's start chatting! I'll be able to ask more insightful questions once we have a conversation going."}), 200

        # Construct the prompt for Gemini
        question_prompt = (
            "Based on the following conversation, generate ONE insightful, open-ended question that encourages the user to "
            "reflect deeper on their feelings, motivations, or situation. The question should be empathetic and "
            "forward-looking, if appropriate. Do NOT summarize or explain. Just the question."
        )
        
        payload = {
            "contents": conversation_context, # Send relevant history as contents
            "system_instruction": {"parts": [{"text": question_prompt}]}, # System instruction for question generation
            "generationConfig": {
                "maxOutputTokens": 50, # Keep the question concise
                "temperature": 0.7,
                "topP": 0.9,
                "topK": 30
            }
        }
        
        headers = {'Content-Type': 'application/json'}
        api_url_with_key = f"{GEMINI_API_URL}?key={GEMINI_API_KEY}" 

        gemini_response = requests.post(api_url_with_key, headers=headers, json=payload)
        gemini_response.raise_for_status()

        gemini_result = gemini_response.json()
        question_text = "Could not generate an insightful question right now."
        if gemini_result and gemini_result.get('candidates') and \
           gemini_result['candidates'][0].get('content') and \
           gemini_result['candidates'][0]['content'].get('parts'):
            question_text = gemini_result['candidates'][0]['content']['parts'][0]['text'].strip()

        return jsonify({"question": question_text}), 200

    except requests.exceptions.RequestException as req_err:
        print(f"❌ Gemini API Request Error for question generation: {req_err}")
        return jsonify({"error": "Failed to connect to AI for a question. Please try again later."}), 500
    except Exception as e:
        print(f"❌ Error during insightful question generation: {e}")
        return jsonify({"error": f"An unexpected error occurred during question generation: {e}"}), 500


if __name__ == '__main__':
    app.run(debug=True)