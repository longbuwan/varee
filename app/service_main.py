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

setting.set_api_key(cfg.AIFORTHAI_APIKEY)
line_bot_api = LineBotApi(cfg.LINE_CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(cfg.LINE_CHANNEL_SECRET)

client = OpenAI(api_key=cfg.OPENAI_APIKEY)


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
    # session id (can be used for AI FOR THAI if needed)
    current_time = datetime.now()
    day, month = current_time.day, current_time.month
    hour, minute = current_time.hour, current_time.minute
    adjusted_minute = minute - (minute % 10)
    session_id = f"{day:02}{month:02}{hour:02}{adjusted_minute:02}"

    # OpenAI GPT-4.1 response
    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=[
            {"role": "user", "content": event.message.text}
        ]
    )

    text = response.choices[0].message.content.strip()
    print("AI raw response:", repr(text))  # ✅ Move print here
    send_message(event, text)


def echo(event):
    line_bot_api.reply_message(
        event.reply_token, TextSendMessage(text=event.message.text)
    )


def send_message(event, message):
    line_bot_api.reply_message(event.reply_token, TextSendMessage(text=message))
