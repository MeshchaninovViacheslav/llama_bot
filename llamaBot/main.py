from assistant import Assistant
from fastapi import FastAPI

ass = Assistant()
app = FastAPI()

@app.post("/message")
def message(user_id: str, message: str):
    ai_message = ass(message)
    return ai_message