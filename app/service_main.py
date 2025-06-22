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
    line_bot_api.reply_message(event.reply_token, TextSendMessage(text=message))

import os

def setup_rich_menu_once():
    flag_file = "rich_menu_setup_flag.txt"
    if os.path.exists(flag_file):
        with open(flag_file, "r") as f:
            status = f.read().strip()
            if status == "done":
                print("Rich menu already set up. Skipping.")
                return




    
    # --- Setup Rich Menu ---
rich_menu_to_create = RichMenu(
        size={"width": 2500, "height": 843},
        selected=True,
        name="My Menu",
        chat_bar_text="Tap here",
        areas=[
            RichMenuArea(
                bounds=RichMenuBounds(x=0, y=0, width=2500, height=843),
                action=URIAction(
                    label="Visit Site",
                    uri="https://vareepri-longbuwans-projects.vercel.app/"
                )
            )
        ]
    )

rich_menu_id = line_bot_api.create_rich_menu(rich_menu=rich_menu_to_create)
line_bot_api.set_default_rich_menu(rich_menu_id)
print("✅ Rich Menu created:", rich_menu_id)

    # Mark as done so it doesn't run again
with open(flag_file, "w") as f:
    f.write("done")

# Call the setup function when the app starts
setup_rich_menu_once()
