from fastapi import APIRouter, Request

from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, TextMessage, TextSendMessage

from datetime import datetime
import openai

from app.configs import Configs

router = APIRouter(tags=["Main"], prefix="/message")

cfg = Configs()

# Set your OpenAI API key
openai.api_key = cfg.OPENAI_API_KEY  # Make sure you add this key to your config

line_bot_api = LineBotApi(cfg.LINE_CHANNEL_ACCESS_TOKEN)  # LINE Channel access token
handler = WebhookHandler(cfg.LINE_CHANNEL_SECRET)  # LINE Channel secret


@router.post("")
async def multimodal_demo(request: Request):
    """
    Line Webhook endpoint for receiving messages from the LINE Messaging API and processing them using OpenAI GPT.
    """
    signature = request.headers["X-Line-Signature"]
    body = await request.body()
    try:
        handler.handle(body.decode("UTF-8"), signature)
    except InvalidSignatureError:
        print("Invalid signature. Please check your channel access token or channel secret.")
    return "OK"


@handler.add(MessageEvent, message=TextMessage)
def handle_text_message(event):
    # session ID (optional)
    now = datetime.now()
    session_id = f"{now.day:02}{now.month:02}{now.hour:02}{now.minute - (now.minute % 10):02}"

    # Call OpenAI ChatGPT
    user_input = event.message.text
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # or "gpt-4" if you have access
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": user_input}
            ],
            temperature=0.6
        )
        reply = response.choices[0].message["content"].strip()
    except Exception as e:
        reply = "เกิดข้อผิดพลาดในการเรียกใช้งาน OpenAI API: " + str(e)

    # Send response back to user
    send_message(event, reply)


# Helper function for sending messages
def send_message(event, message):
    line_bot_api.reply_message(
        event.reply_token,
        TextSendMessage(text=message)
    )
    print("AI raw response:", repr(message))
