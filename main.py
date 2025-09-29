import asyncio
from audio_listener import KeywordListener
from rasa_sdk.executor import CollectingDispatcher
from rasa.core.agent import Agent
import os

async def send_to_rasa(text):
    # Aquí podés enviar el texto a tu bot Rasa
    print(f"➡ Sending to Rasa: {text}")
    agent = await Agent.load("models")  # Asegurate que tu ruta a models sea correcta
    responses = await agent.handle_text(text)
    for response in responses:
        print(f"Rasa response: {response.get('text')}")

async def main():
    listener = KeywordListener(callback=send_to_rasa)
    await listener.start_listening()

if __name__ == "__main__":
    asyncio.run(main())
