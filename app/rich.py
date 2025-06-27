from linebot import LineBotApi
from linebot.models import RichMenu, RichMenuArea, RichMenuBounds, URIAction
from app.configs import Configs

cfg = Configs()
line_bot_api = LineBotApi(cfg.LINE_CHANNEL_ACCESS_TOKEN)

# Create the rich menu
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

# Upload the menu
rich_menu_id = line_bot_api.create_rich_menu(rich_menu=rich_menu_to_create)
print("Created Rich Menu ID:", rich_menu_id)

# OPTIONAL: Upload image if you want
# with open("static/images/menu.png", "rb") as f:
#     line_bot_api.set_rich_menu_image(rich_menu_id, "image/png", f)

# Set as default
line_bot_api.set_default_rich_menu(rich_menu_id)
