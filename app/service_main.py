from fastapi import APIRouter, Request

from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, TextMessage, TextSendMessage

from aift import setting
from aift.multimodal import textqa
from openai import OpenAI
from datetime import datetime

from app.configs import Configs

router = APIRouter(tags=["Main"], prefix="/message")

cfg = Configs()

setting.set_api_key(cfg.AIFORTHAI_APIKEY) # AIFORTHAI_APIKEY
line_bot_api = LineBotApi(cfg.LINE_CHANNEL_ACCESS_TOKEN)  # CHANNEL_ACCESS_TOKEN
handler = WebhookHandler(cfg.LINE_CHANNEL_SECRET)  # CHANNEL_SECRET

lient = OpenAI(api_key=cfg.OPENAI_APIKEY)
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
    signature = request.headers["X-Line-Signature"]
    body = await request.body()
    try:
        handler.handle(body.decode("UTF-8"), signature)
    except InvalidSignatureError:
        print(
            "Invalid signature. Please check your channel access token or channel secret."
        )
    return "OK"


@handler.add(MessageEvent, message=TextMessage)
def handle_text_message(event):
    # session id
    current_time = datetime.now()
    # extract day, month, hour, and minute
    day, month = current_time.day, current_time.month
    hour, minute = current_time.hour, current_time.minute
    # adjust the minute to the nearest lower number divisible by 10
    adjusted_minute = minute - (minute % 10)
    result = f"{day:02}{month:02}{hour:02}{adjusted_minute:02}"

    # aiforthai multimodal chat
    #text = textqa.chat(
    #    event.message.text, result + cfg.AIFORTHAI_APIKEY, temperature=0.6, context=""
    #)["response"]
    text = client.responses.create(
        model = "gpt-4.1",
       input = event.message.text
   
    )
    # return text response
    send_message(event, text.output.text)


def echo(event):
    line_bot_api.reply_message(
        event.reply_token, TextSendMessage(text=event.message.text)
    )


# function for sending message
def send_message(event, message):
    line_bot_api.reply_message(event.reply_token, TextSendMessage(text=message))

    print("AI raw response:", repr(text))
