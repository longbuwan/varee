from fastapi import APIRouter, Request, HTTPException
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError, LineBotApiError
from linebot.models import MessageEvent, TextMessage, TextSendMessage

from aift import setting
from aift.multimodal import textqa


from datetime import datetime
from app.configs import Configs
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter(tags=["Main"], prefix="/message")

cfg = Configs()

# Set AI FOR THAI API key
setting.set_api_key(cfg.AIFORTHAI_APIKEY)

# Initialize LINE Bot API
line_bot_api = LineBotApi(cfg.LINE_CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(cfg.LINE_CHANNEL_SECRET)

@router.post("/")
async def multimodal_demo(request: Request):
    """
    Line Webhook endpoint สำหรับรับข้อความจาก Line Messaging API และประมวลผลข้อความด้วย AI FOR THAI 
    
    ฟังก์ชันนี้ทำหน้าที่:
    1. รับ HTTP POST Request จาก Line Webhook
    2. ตรวจสอบลายเซ็น (X-Line-Signature) เพื่อยืนยันความถูกต้องของข้อความ
    3. ส่งข้อความไปยัง handler เพื่อประมวลผลอีเวนต์ที่ได้รับ
    4. เมื่อได้รับข้อความ (MessageEvent) ที่เป็นข้อความ (TextMessage):
        - สร้าง session id โดยใช้วัน, เดือน, ชั่วโมง, และนาทีที่ปรับให้ลงตัวกับเลข 10
        - ส่งข้อความไปยัง API Text QA ของ AI FOR THAI (ซึ่งใช้ Pathumma LLM) เพื่อประมวลผล
        - ส่งข้อความตอบกลับ (response) กลับไปยังผู้ใช้ผ่าน Line Messaging API
    """
    try:
        # Get signature header
        signature = request.headers.get("X-Line-Signature")
        if not signature:
            logger.error("Missing X-Line-Signature header")
            raise HTTPException(status_code=400, detail="Missing signature")
        
        # Get request body
        body = await request.body()
        body_str = body.decode("UTF-8")
        
        logger.info(f"Received webhook request with signature: {signature}")
        logger.info(f"Request body: {body_str}")
        
        # Handle the webhook
        handler.handle(body_str, signature)
        
    except InvalidSignatureError:
        logger.error("Invalid signature. Please check your channel access token or channel secret.")
        raise HTTPException(status_code=400, detail="Invalid signature")
    except Exception as e:
        logger.error(f"Error processing webhook: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")
    
    return {"status": "OK"}

@handler.add(MessageEvent, message=TextMessage)
def handle_text_message(event):
    """Handle incoming text messages"""
    try:
        user_message = event.message.text
        user_id = event.source.user_id
        
        logger.info(f"Received message from user {user_id}: {user_message}")
        
        # Create session id
        current_time = datetime.now()
        day, month = current_time.day, current_time.month
        hour, minute = current_time.hour, current_time.minute
        
        # Adjust the minute to the nearest lower number divisible by 10
        adjusted_minute = minute - (minute % 10)
        session_id = f"{day:02}{month:02}{hour:02}{adjusted_minute:02}"
        
        logger.info(f"Generated session ID: {session_id}")
        
        # Call AI FOR THAI API
        try:
            full_session_id = session_id + cfg.AIFORTHAI_APIKEY
            logger.info(f"Calling textqa.chat with session_id: {session_id}")
            
            ai_response = textqa.chat(
                user_message, 
                full_session_id, 
                temperature=0.6, 
                context=""
            )
            
            logger.info(f"AI response object: {ai_response}")
            
            # Extract response text
            if isinstance(ai_response, dict) and "response" in ai_response:
                response_text = ai_response["response"]
            else:
                response_text = str(ai_response)
            
            logger.info(f"AI raw response: {repr(response_text)}")
            
            # Check if response is empty or None
            if not response_text or response_text.strip() == "":
                response_text = "ขออภัย ไม่สามารถประมวลผลคำถามของคุณได้ในขณะนี้"
            
            # Send response back to user
            send_message(event, response_text)
            
        except Exception as ai_error:
            logger.error(f"AI API error: {str(ai_error)}")
            # Send fallback message
            fallback_message = "ขออภัย เกิดข้อผิดพลาดในการประมวลผล กรุณาลองใหม่อีกครั้ง"
            send_message(event, fallback_message)
            
    except Exception as e:
        logger.error(f"Error in handle_text_message: {str(e)}")
        try:
            # Send error message to user
            error_message = "ขออภัย เกิดข้อผิดพลาดของระบบ กรุณาลองใหม่ภายหลัง"
            send_message(event, error_message)
        except:
            logger.error("Failed to send error message to user")

def send_message(event, message):
    """Function for sending message back to LINE"""
    try:
        logger.info(f"Sending message to user: {repr(message)}")
        
        # Ensure message is not empty and is a string
        if not message or not isinstance(message, str):
            message = "ขออภัย ไม่สามารถสร้างข้อความตอบกลับได้"
        
        # Truncate message if too long (LINE has a 5000 character limit)
        if len(message) > 5000:
            message = message[:4900] + "...\n(ข้อความยาวเกินไป)"
        
        line_bot_api.reply_message(
            event.reply_token, 
            TextSendMessage(text=message)
        )
        
        logger.info("Message sent successfully")
        
    except LineBotApiError as e:
        logger.error(f"LINE Bot API error: {e.status_code} - {e.error.message}")
    except Exception as e:
        logger.error(f"Error sending message: {str(e)}")

def echo(event):
    """Simple echo function for testing"""
    try:
        line_bot_api.reply_message(
            event.reply_token, 
            TextSendMessage(text=event.message.text)
        )
    except Exception as e:
        logger.error(f"Error in echo: {str(e)}")

# Health check endpoint
@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "Thai LINE Bot with AI FOR THAI"
    }
