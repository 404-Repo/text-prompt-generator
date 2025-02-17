from discord_webhook import DiscordWebhook
from loguru import logger
from pydantic import HttpUrl


def post_exception_to_discord(e: Exception, prompt_generator_number: str, webhook_url: HttpUrl) -> None:
    discord_message = (
        f"🚨 **Server Error Notification, PROMPT GENERATOR {prompt_generator_number}** 🚨\n\n" f"**Error:** {str(e)}\n"
    )
    send_discord_message(discord_message, webhook_url)


def send_discord_message(message: str, webhook_url: HttpUrl) -> None:
    webhook = DiscordWebhook(url=str(webhook_url), content=message)
    response = webhook.execute()
    if response.status_code != 200:
        logger.error(f"Failed to send message to Discord. Status code: {response.status_code}")
