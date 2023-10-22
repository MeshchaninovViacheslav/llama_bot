# from langchain.llms import LlamaCpp
# from langchain.prompts import PromptTemplate
# from langchain.chains import LLMChain
# from langchain.callbacks.manager import CallbackManager
# from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# # Sources
# # https://huggingface.co/TheBloke/Llama-2-7b-Chat-GGUF


# # Make sure the model path is correct for your system!
# llm = LlamaCpp(
#     model_path="./llama-2-7b-chat.Q4_K_M.gguf",
#     temperature=0.75,
#     max_tokens=500,
#     top_p=1,
#     verbose=False, # Verbose is required to pass to the callback manager
# )


# template = """
# System:
# Вы услужливый, уважительный и честный ассистент банка Тинькофф. 
# Всегда отвечайте как можно более полезно и безопасно. 
# Ваши ответы не должны содержать никакого вредного, неэтичного, расистского, сексистского, токсичного, опасного или незаконного контента. 
# Пожалуйста, убедитесь, что ваши ответы носят социально непредвзятый и позитивный характер. 
# Если вопрос не имеет никакого смысла или фактически не согласован, объясните почему, вместо того чтобы отвечать на что-то неправильное. 
# Если вы не знаете ответа на вопрос, пожалуйста, не распространяйте ложную информацию.
# Отвечайте только на русском языке.
# User: {prompt}
# Assistant:
# """

# prompt = PromptTemplate(template=template, input_variables=["prompt"])
# llm_chain = LLMChain(prompt=prompt, llm=llm)
# question = """Сколько мне будут стоить смски с оповещениями об операциях?"""
# answer = llm_chain.run(question)
# print(answer)
from time import time
from assistant import Assistant

ass = Assistant()

t1 = time()
question = """Сколько мне будут стоить смски с оповещениями об операциях?"""
print(ass(question))
print(time() - t1)

from fastapi import FastAPI

app = FastAPI()

@app.post("/message")
def 
