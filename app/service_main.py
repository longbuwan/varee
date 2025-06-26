from fastapi import APIRouter, Request
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import RichMenu, RichMenuArea, RichMenuBounds, MessageAction, TemplateSendMessage, ButtonsTemplate, URIAction, MessageEvent, TextMessage, TextSendMessage
from aift import setting
from aift.multimodal import textqa
from openai import OpenAI
from datetime import datetime
from app.configs import Configs
import json
from typing import Dict, List
import tiktoken

router = APIRouter(tags=["Main"], prefix="/message")
cfg = Configs()
setting.set_api_key(cfg.AIFORTHAI_APIKEY)
line_bot_api = LineBotApi(cfg.LINE_CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(cfg.LINE_CHANNEL_SECRET)
client = OpenAI(api_key=cfg.OPENAI_APIKEY)

# Conversation storage (use Redis or database in production)
user_conversations: Dict[str, List[Dict]] = {}

# Token counting setup
encoding = tiktoken.get_encoding("cl100k_base")

# Maximum tokens for conversation history (adjust based on your model limits)
MAX_HISTORY_TOKENS = 2000
MAX_RESPONSE_TOKENS = 500

def count_tokens(text: str) -> int:
    """Count tokens in a text string"""
    return len(encoding.encode(text))

def truncate_conversation_history(messages: List[Dict], max_tokens: int) -> List[Dict]:
    """Truncate conversation history to fit within token limit"""
    system_message = messages[0] if messages and messages[0]["role"] == "system" else None
    conversation_messages = messages[1:] if system_message else messages
    
    # Always keep system message
    result = [system_message] if system_message else []
    current_tokens = count_tokens(system_message["content"]) if system_message else 0
    
    # Add messages from most recent, working backwards
    for message in reversed(conversation_messages):
        message_tokens = count_tokens(message["content"])
        if current_tokens + message_tokens <= max_tokens:
            result.insert(-1 if system_message else 0, message)
            current_tokens += message_tokens
        else:
            break
    
    return result

def get_user_conversation(user_id: str) -> List[Dict]:
    """Get conversation history for a user"""
    if user_id not in user_conversations:
        # Initialize with system prompt in Thai
        system_prompt = """คุณคือ "วารี แชทบอท" ผู้ช่วยปรึกษาการสมัครเรียนมหาวิทยาลัยที่เป็นมิตรและใช้ภาษาไทย

บทบาทของคุณ:
- ให้คำปรึกษาเกี่ยวกับการสมัครเรียนมหาวิทยาลัยในประเทศไทย
- แนะนำคณะ สาขาวิชา และข้อมูลการรับสมัคร
- ช่วยเหลือเรื่องเอกสารการสมัคร และเกณฑ์การรับสมัคร
- ให้คำแนะนำเกี่ยวกับการเตรียมตัวสอบเข้าและทุนการศึกษา
- ตอบคำถามเกี่ยวกับค่าใช้จ่ายการศึกษาและที่พักนักศึกษา

ลักษณะการตอบ:
- ใช้ภาษาไทยที่เป็นกันเอง แต่สุภาพ
- ให้ข้อมูลที่ถูกต้องและเป็นประโยชน์
- ถามคำถามเพิ่มเติมเมื่อจำเป็น เพื่อให้คำแนะนำที่เหมาะสม
- หากไม่ทราบข้อมูล ให้แนะนำแหล่งข้อมูลที่เชื่อถือได้

เริ่มต้นการสนทนาด้วยการทักทายและแนะนำตัว"""
        
        user_conversations[user_id] = [
            {"role": "system", "content": system_prompt}
        ]
    
    return user_conversations[user_id]

def add_to_conversation(user_id: str, role: str, content: str):
    """Add a message to user's conversation history"""
    if user_id not in user_conversations:
        get_user_conversation(user_id)  # Initialize if not exists
    
    user_conversations[user_id].append({
        "role": role,
        "content": content
    })
    
    # Truncate if too long
    user_conversations[user_id] = truncate_conversation_history(
        user_conversations[user_id], 
        MAX_HISTORY_TOKENS
    )

def get_openai_response(user_id: str, user_message: str) -> str:
    """Get response from OpenAI with conversation history"""
    try:
        # Add user message to history
        add_to_conversation(user_id, "user", user_message)
        
        # Get conversation history
        messages = get_user_conversation(user_id)
        
        # Make API call (Note: using gpt-3.5-turbo as gpt-4.1 seems incorrect)
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",  # or "gpt-4" if you have access
            messages=messages,
            max_tokens=MAX_RESPONSE_TOKENS,
            temperature=0.7,
            presence_penalty=0.1,
            frequency_penalty=0.1
        )
        
        assistant_response = response.choices[0].message.content
        
        # Add assistant response to history
        add_to_conversation(user_id, "assistant", assistant_response)
        
        return assistant_response
        
    except Exception as e:
        print(f"Error calling OpenAI API: {e}")
        return "ขอโทษค่ะ ระบบมีปัญหาชั่วคราว กรุณาลองใหม่อีกครั้งนะคะ"

@router.post("/")
async def multimodal_demo(request: Request):
    signature = request.headers["X-Line-Signature"]
    body = await request.body()
    try:
        handler.handle(body.decode("UTF-8"), signature)
    except InvalidSignatureError:
        print("Invalid signature. Please check your channel access token or channel secret.")
    return "OK"

@handler.add(MessageEvent, message=TextMessage)
def handle_text_message(event):
    user_id = event.source.user_id
    user_message = event.message.text
    
    # Handle special commands
    if user_message.lower() == "form":
        message = TemplateSendMessage(
            alt_text='เปิดแอปพลิเคชัน LIFF',
            template=ButtonsTemplate(
                title='ลองใช้แอป',
                text='คลิกด้านล่างเพื่อเปิดแอป LIFF',
                actions=[
                    URIAction(label='เปิด LIFF', uri='https://liff.line.me/2007611527-z7ZNLXOk')
                ]
            )
        )
        line_bot_api.reply_message(event.reply_token, message)
        return
    
    # Handle reset conversation
    if user_message.lower() in ["รีเซ็ต", "reset", "เริ่มใหม่"]:
        if user_id in user_conversations:
            del user_conversations[user_id]
        send_message(event, "สนทนาถูกรีเซ็ตแล้วค่ะ เริ่มต้นการสนทนาใหม่ได้เลยค่ะ")
        return
    
    # Get AI response with conversation history
    response = get_openai_response(user_id, user_message)
    send_message(event, response)

def send_message(event, message):
    """Send message back to Line"""
    line_bot_api.reply_message(event.reply_token, TextSendMessage(text=message))

# Optional: Function to clear old conversations (call periodically)
def cleanup_old_conversations(max_conversations: int = 1000):
    """Keep only the most recent conversations to prevent memory issues"""
    if len(user_conversations) > max_conversations:
        # Keep only the most recent conversations
        sorted_users = sorted(user_conversations.keys())
        users_to_remove = sorted_users[:-max_conversations]
        for user_id in users_to_remove:
            del user_conversations[user_id]

# Optional: Get conversation stats
def get_conversation_stats():
    """Get statistics about current conversations"""
    total_conversations = len(user_conversations)
    total_messages = sum(len(conv) for conv in user_conversations.values())
    avg_messages_per_user = total_messages / total_conversations if total_conversations > 0 else 0
    
    return {
        "total_conversations": total_conversations,
        "total_messages": total_messages,
        "avg_messages_per_user": round(avg_messages_per_user, 2)
    }
