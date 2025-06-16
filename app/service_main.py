from fastapi import APIRouter, Request

from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, TextMessage, TextSendMessage

from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

from app.configs import Configs

router = APIRouter(tags=["Main"], prefix="/message")

cfg = Configs()

line_bot_api = LineBotApi(cfg.LINE_CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(cfg.LINE_CHANNEL_SECRET)

# Load Typhoon2 model and tokenizer once (assumes 7B-chat model)
print("Loading Typhoon2 model... This may take a while.")
tokenizer = AutoTokenizer.from_pretrained("scb10x/typhoon2-7b-chat")
model = AutoModelForCausalLM.from_pretrained("scb10x/typhoon2-7b-chat", torch_dtype=torch.float16, device_map="auto")
model.eval()
print("Typhoon2 model loaded.")


@router.post("")
async def multimodal_demo(request: Request):
    """
    Line Webhook endpoint for receiving messages from the LINE Messaging API and processing them using Typhoon2 LLM.
    """
    signature = request.headers.get("X-Line-Signature", "")
    body = await request.body()

    try:
        handler.handle(body.decode("UTF-8"), signature)
    except InvalidSignatureError:
        print("Invalid signature. Please check your channel access token or channel secret.")
        return "Invalid signature", 400
    return "OK"


@handler.add(MessageEvent, message=TextMessage)
def handle_text_message(event):
    now = datetime.now()
    session_id = f"{now.day:02}{now.month:02}{now.hour:02}{now.minute - (now.minute % 10):02}"

    user_input = event.message.text

    try:
        reply = generate_typhoon2_reply(user_input)
    except Exception as e:
        reply = "เกิดข้อผิดพลาดในการเรียกใช้งาน Typhoon2 โมเดล: " + str(e)

    send_message(event, reply)


def generate_typhoon2_reply(user_input: str) -> str:
    """
    Generate reply from Typhoon2 7B chat model.
    """
    # Prepare input prompt with system role for chat style
    system_prompt = "You are a helpful assistant."
    prompt = f"{system_prompt}\nUser: {user_input}\nAssistant:"

    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)

    # Generate response
    output_ids = model.generate(
        input_ids,
        max_length=512,
        do_sample=True,
        temperature=0.6,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        no_repeat_ngram_size=3,
        num_return_sequences=1
    )

    # Decode generated tokens (skip prompt tokens)
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    # Remove the prompt part from generated text to get only the assistant's reply
    reply = generated_text[len(prompt):].strip()
    return reply


def send_message(event, message):
    line_bot_api.reply_message(
        event.reply_token,
        TextSendMessage(text=message)
    )
    print("AI raw response:", repr(message))
