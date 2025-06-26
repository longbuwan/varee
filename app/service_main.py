
from fastapi import APIRouter, Request
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import RichMenu, RichMenuArea, RichMenuBounds, MessageAction, TemplateSendMessage, ButtonsTemplate, URIAction, MessageEvent, TextMessage, TextSendMessage
from aift import setting
from aift.multimodal import textqa
from openai import OpenAI
from datetime import datetime
from app.configs import Configs
import tiktoken
from typing import Dict, List

router = APIRouter(tags=["Main"], prefix="/message")
cfg = Configs()
setting.set_api_key(cfg.AIFORTHAI_APIKEY)
line_bot_api = LineBotApi(cfg.LINE_CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(cfg.LINE_CHANNEL_SECRET)
client = OpenAI(api_key=cfg.OPENAI_APIKEY)

# In-memory storage for conversation history (use database for production)
user_conversations: Dict[str, List[Dict[str, str]]] = {}

# Token counting for optimization
encoding = tiktoken.encoding_for_model("gpt-4")

# Maximum tokens configuration
MAX_CONTEXT_TOKENS = 2000  # Leave room for response
MAX_CONVERSATION_MESSAGES = 20  # Maximum messages to keep per user

# Enhanced Thai system prompt
SYSTEM_PROMPT = """คุณคือ "วารี แชทบอท" ผู้ช่วยอัจฉริยะสำหรับการให้คำแนะนำเกี่ยวกับการสมัครเรียนมหาวิทยาลัย

บทบาทและหน้าที่ของคุณ:
- ให้คำแนะนำที่เป็นประโยชน์เกี่ยวกับการสมัครเรียนในมหาวิทยาลัยต่างๆ
- ตอบคำถามเกี่ยวกับหลักสูตร คุณสมบัติผู้สมัคร และขั้นตอนการสมัคร
- แนะนำเกี่ยวกับการเตรียมตัวสอบเข้า และเอกสารที่จำเป็น
- ให้ข้อมูลเกี่ยวกับทุนการศึกษาและโอกาสต่างๆ

คำแนะนำในการตอบ:
- ตอบเป็นภาษาไทยที่เข้าใจง่าย
- ให้ข้อมูลที่ถูกต้องและเป็นประโยชน์
- หากไม่แน่ใจ ให้แนะนำให้ติดต่อสถาบันโดยตรง
- มีมารยาทและเป็นมิตรเสมอ
- ให้คำตอบที่กระชับแต่ครบถ้วน

จำไว้: คุณคือผู้ช่วยที่เชี่ยวชาญด้านการศึกษาและพร้อมช่วยเหลือนักเรียนในการวางแผนอนาคต"""

def count_tokens(messages: List[Dict[str, str]]) -> int:
    """Count tokens in message list"""
    text = ""
    for msg in messages:
        text += f"{msg['role']}: {msg['content']}\n"
    return len(encoding.encode(text))

def manage_conversation_history(user_id: str, new_message: str) -> List[Dict[str, str]]:
    """Manage conversation history with token limits"""
    if user_id not in user_conversations:
        user_conversations[user_id] = []
    
    # Add new user message
    user_conversations[user_id].append({
        "role": "user",
        "content": new_message
    })
    
    # Keep only recent messages if too many
    if len(user_conversations[user_id]) > MAX_CONVERSATION_MESSAGES:
        user_conversations[user_id] = user_conversations[user_id][-MAX_CONVERSATION_MESSAGES:]
    
    # Create messages for API call
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(user_conversations[user_id])
    
    # Trim messages if token count is too high
    while count_tokens(messages) > MAX_CONTEXT_TOKENS and len(messages) > 2:
        # Remove oldest user-assistant pair (keep system message)
        if len(messages) > 3:
            messages.pop(1)  # Remove oldest user message
            if len(messages) > 2 and messages[1]["role"] == "assistant":
                messages.pop(1)  # Remove corresponding assistant message
        else:
            break
    
    return messages

def add_assistant_response(user_id: str, response: str):
    """Add assistant response to conversation history"""
    if user_id in user_conversations:
        user_conversations[user_id].append({
            "role": "assistant", 
            "content": response
        })

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
    
    # Handle clear conversation command
    if user_message.lower() in ["clear", "เคลียร์", "ล้างประวัติ"]:
        if user_id in user_conversations:
            del user_conversations[user_id]
        send_message(event, "ล้างประวัติการสนทนาเรียบร้อยแล้ว! พร้อมเริ่มการสนทนาใหม่ 😊")
        return
    
    try:
        # Get conversation history with token management
        messages = manage_conversation_history(user_id, user_message)
        
        # Build conversation context using your original format
        conversation_context = SYSTEM_PROMPT + "\n\n"
        
        # Add conversation history (excluding system message and current message)
        for msg in user_conversations[user_id][:-1]:  # All except the last (current) message
            role_thai = "ผู้ใช้" if msg["role"] == "user" else "ผู้ช่วย"
            conversation_context += f"{role_thai}: {msg['content']}\n"
        
        # Add current message
        conversation_context += f"ผู้ใช้: {user_message}"
        
        # Use your original API call format that was working
        response = client.responses.create(
            model="gpt-4.1",
            input=conversation_context
        )
        
        assistant_response = response.output_text
        
        # Add assistant response to history
        add_assistant_response(user_id, assistant_response)
        
        # Send response
        send_message(event, assistant_response)
        
    except Exception as e:
        print(f"Error calling OpenAI API: {e}")
        error_message = "ขออภัย เกิดข้อผิดพลาดในการประมวลผล กรุณาลองใหม่อีกครั้ง"
        send_message(event, error_message)

def send_message(event, message):
    """Send message with length check"""
    # LINE has a 5000 character limit per message
    if len(message) > 4800:
        # Split long messages
        chunks = [message[i:i+4800] for i in range(0, len(message), 4800)]
        for i, chunk in enumerate(chunks):
            if i == 0:
                line_bot_api.reply_message(event.reply_token, TextSendMessage(text=chunk))
            else:
                line_bot_api.push_message(event.source.user_id, TextSendMessage(text=chunk))
    else:
        line_bot_api.reply_message(event.reply_token, TextSendMessage(text=message))

# Optional: Function to get conversation statistics
def get_conversation_stats(user_id: str) -> Dict:
    """Get conversation statistics for a user"""
    if user_id not in user_conversations:
        return {"message_count": 0, "token_estimate": 0}
    
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(user_conversations[user_id])
    
    return {
        "message_count": len(user_conversations[user_id]),
        "token_estimate": count_tokens(messages)
    }

# Optional: Function to export conversation history
def export_conversation(user_id: str) -> str:
    """Export conversation history for a user"""
    if user_id not in user_conversations:
        return "ไม่มีประวัติการสนทนา"
    
    conversation = ""
    for msg in user_conversations[user_id]:
        role = "ผู้ใช้" if msg["role"] == "user" else "ผู้ช่วย"
        conversation += f"{role}: {msg['content']}\n\n"
    
    return conversation
