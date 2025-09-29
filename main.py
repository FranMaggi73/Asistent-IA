# main.py (sin cambios grandes)
import asyncio
import aiohttp
from audio_listener import KeywordListener
from dotenv import load_dotenv
import os

load_dotenv()
RASA_URL = os.getenv("RASA_URL")
assert RASA_URL is not None, "❌ RASA_URL is not set in the .env file"

async def send_to_rasa(text: str):
    print(f"➡ Sending to Rasa: {text}")
    async with aiohttp.ClientSession() as session:
        payload = {"sender": "user", "message": text}
        async with session.post(RASA_URL, json=payload) as resp:
            print("HTTP status:", resp.status)
            responses = await resp.json()
            print("Responses from Rasa:", responses)
            for response in responses:
                print(f"🤖 Rasa: {response.get('text')}")
    # 🔁 Después de responder, volvemos a escuchar
    print("✅ Listening for 'Jarvis' again...")

async def main():
    listener = KeywordListener(callback=send_to_rasa)
    print("🟢 Assistant is ready. Say 'Jarvis' to start speaking...")
    await listener.start_listening()

if __name__ == "__main__":
    asyncio.run(main())
