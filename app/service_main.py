from fastapi import APIRouter, Request
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import RichMenu, RichMenuArea, RichMenuBounds, MessageAction,TemplateSendMessage, ButtonsTemplate, URIAction, MessageEvent, TextMessage, TextSendMessage
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
context_info = (
    "Chat name: Varee Chat Bot\n"
    "Assistant: Helpful AI that gives University enrollment tips\n"
)
@handler.add(MessageEvent, message=TextMessage)
def handle_text_message(event):
    # session id (can be used for AI FOR THAI if needed)
    current_time = datetime.now()
    day, month = current_time.day, current_time.month
    hour, minute = current_time.hour, current_time.minute
    adjusted_minute = minute - (minute % 10)
    session_id = f"{day:02}{month:02}{hour:02}{adjusted_minute:02}"
    # OpenAI GPT-4.1 response
    if event.message.text != "form":
        response = client.responses.create(
            model="gpt-4.1",
            input= context_info + event.message.text
        )
        send_message(event, response.output_text)
    else:

        message = TemplateSendMessage(
            alt_text='Open the LIFF App',
            template=ButtonsTemplate(
                title='Try the App',
                text='Click below to open the LIFF app.',
                actions=[
                    URIAction(label='Open LIFF', uri='https://liff.line.me/2007611527-z7ZNLXOk')
                ]
            )
        )
        line_bot_api.reply_message(event.reply_token, message)
def echo(event):
    line_bot_api.reply_message(
        event.reply_token, TextSendMessage(text=event.message.text)
    )
def send_message(event, message):
    line_bot_api.reply_message(event.reply_token, TextSendMessage(text=message)) can you make this chat bot keep the old messages data of each user and set a better prompt for it in thai also optimise the token usage so i dont waste too much token

