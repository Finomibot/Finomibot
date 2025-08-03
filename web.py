
import logging
import pickle
from fastapi import FastAPI, Request
from telegram import Update, Bot
from sklearn.metrics.pairwise import cosine_similarity
import uvicorn

TELEGRAM_TOKEN = "7950081149:AAEwHty0BGgWNc-VKOxBBkVLoJNH6cE9QD8"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Carica dataset, vectorizer e vettori
with open("database_pillole_finanza.pkl", "rb") as f:
    data = pickle.load(f)

df = data["data"]
vectorizer = data["vectorizer"]
X = data["vectors"]

bot = Bot(token=TELEGRAM_TOKEN)
app = FastAPI()

def cerca_pillola(domanda):
    try:
        domanda_vect = vectorizer.transform([domanda])
        sim = cosine_similarity(domanda_vect, X)[0]
        top_idx = sim.argmax()
        top_pillola = df.iloc[top_idx]
        return top_pillola["titolo"], top_pillola["fonte"], top_pillola["testo"]
    except Exception as e:
        logger.error(f"Errore nella ricerca della pillola: {e}")
        return "Errore", "Sistema", "Non è stato possibile generare una risposta."

@app.post("/")
async def webhook(request: Request):
    try:
        data = await request.json()
        update = Update.de_json(data, bot)
        message = update.message.text if update.message else None
        chat_id = update.message.chat_id if update.message else None

        if message and chat_id:
            titolo, fonte, contenuto = cerca_pillola(message)
            risposta = f"📘 *{titolo}*\n{contenuto}\n\n🔗 Fonte: {fonte}"
            await bot.send_message(chat_id=chat_id, text=risposta, parse_mode="Markdown")
        return {"status": "ok"}
    except Exception as e:
        logger.error(f"Errore nella gestione del webhook: {e}")
        return {"status": "error", "message": str(e)}

@app.get("/")
async def root():
    return {"message": "Finomibot is running"}

if __name__ == "__main__":
    uvicorn.run("web:app", host="0.0.0.0", port=10000)
