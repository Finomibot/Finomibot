
import pickle
import logging
from fastapi import FastAPI, Request
import uvicorn
from telegram import Update, Bot
from sklearn.metrics.pairwise import cosine_similarity

TELEGRAM_TOKEN = "7950081149:AAEwHty0BGgWNc-VKOxBBkVLoJNH6cE9QD8"

with open("database_pillole_finanza.pkl", "rb") as f:
    data = pickle.load(f)

df = data["data"]
vectorizer = data["vectorizer"]

bot = Bot(token=TELEGRAM_TOKEN)
app = FastAPI()

def cerca_pillola(domanda):
    domanda_vect = vectorizer.transform([domanda])
    vectors = df.drop(columns=["titolo", "fonte"]).values
    sim = cosine_similarity(domanda_vect, vectors)[0]
    top_idx = sim.argmax()
    top_pillola = df.iloc[top_idx]
    return top_pillola["titolo"], top_pillola["fonte"], sim[top_idx]

def genera_risposta(domanda, titolo, fonte):
    contenuto = df[df["titolo"] == titolo].drop(columns=["titolo", "fonte"]).values[0]
    risposta = (
        f"📘 *{titolo}*\n"
        f"{' '.join(list(contenuto)[:50])}...\n\n"
        f"✅ Fonte: {fonte}\n"
        f"📩 Domanda ricevuta: _{domanda}_"
    )
    return risposta

@app.post("/")
async def webhook(request: Request):
    data = await request.json()
    update = Update.de_json(data, bot)
    message = update.message.text.strip()
    chat_id = update.message.chat.id

    if message.startswith("/analizza"):
        ticker = message.replace("/analizza", "").strip().upper()
        risposta = f"🔍 Analisi su {ticker}: (modulo operativo in arrivo)"
    else:
        titolo, fonte, sim = cerca_pillola(message)
        risposta = genera_risposta(message, titolo, fonte)

    await bot.send_message(chat_id=chat_id, text=risposta, parse_mode="Markdown")
    return {"status": "ok"}

@app.get("/")
async def root():
    return {"message": "Finomibot is running"}

if __name__ == "__main__":
    uvicorn.run("web:app", host="0.0.0.0", port=10000)
