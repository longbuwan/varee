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
@router.get("/")
async def health_check():
    return {"status": "ok", "message": "LINE bot service is running."}

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
        # Build conversation input with memory
        conversation_input = SYSTEM_PROMPT + "\n\n"
        
        # Add conversation history if exists
        if user_id in user_conversations and user_conversations[user_id]:
            # Only include last 6 messages to save tokens (3 exchanges)
            recent_messages = user_conversations[user_id][-6:]
            for msg in recent_messages:
                role_thai = "ผู้ใช้" if msg["role"] == "user" else "ผู้ช่วย"
                conversation_input += f"{role_thai}: {msg['content']}\n"
        
        # Add current message (same format as your original)
        conversation_input += user_message
        
        # Try different possible API formats for gpt-4.1
        assistant_response = None
        
        # Method 1: Your original format
        try:
            response = client.responses.create(
                model="gpt-4.1",
                input=conversation_input
            )
            assistant_response = response.output_text
            print("Success with responses.create method")
        except AttributeError:
            print("responses.create not available, trying alternatives...")
            
            # Method 2: Standard OpenAI with gpt-4.1
            try:
                response = client.chat.completions.create(
                    model="gpt-4.1",
                    messages=[{"role": "user", "content": conversation_input}]
                )
                assistant_response = response.choices[0].message.content
                print("Success with chat.completions.create method")
            except Exception as e:
                print(f"chat.completions.create failed: {e}")
                
                # Method 3: Direct completions
                try:
                    response = client.completions.create(
                        model="gpt-4.1",
                        prompt=conversation_input,
                        max_tokens=500
                    )
                    assistant_response = response.choices[0].text
                    print("Success with completions.create method")
                except Exception as e:
                    print(f"completions.create failed: {e}")
                    
                    # Method 4: Check what methods are actually available
                    available_methods = [method for method in dir(client) if not method.startswith('_')]
                    print(f"Available client methods: {available_methods}")
                    raise Exception("No working API method found for gpt-4.1")
        
        if assistant_response:
            # Add to conversation history
            if user_id not in user_conversations:
                user_conversations[user_id] = []
            
            user_conversations[user_id].append({"role": "user", "content": user_message})
            user_conversations[user_id].append({"role": "assistant", "content": assistant_response})
            
            # Keep only last 20 messages per user
            if len(user_conversations[user_id]) > 20:
                user_conversations[user_id] = user_conversations[user_id][-20:]
            
            # Send response
            send_message(event, assistant_response)
        
    except Exception as e:
        print(f"All API methods failed. Error: {type(e).__name__}: {str(e)}")
        print("Please check your OpenAI client setup or provide the correct API format")
        
        # Fallback: Use original code without memory as last resort
        try:
            print("Using original fallback without conversation memory...")
            # Try to replicate your exact original working code
            response = client.responses.create(
                model="gpt-4.1",
                input=SYSTEM_PROMPT + "\n\n" + user_message
            )
            send_message(event, response.output_text)
        except:
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
