import torch

from langchain.llms import LlamaCpp
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

from sentence_encoder import SentenceEncoder
from index import Index

# Sources
# https://huggingface.co/TheBloke/Llama-2-7b-Chat-GGUF
# https://github.com/abetlen/llama-cpp-python/blob/main/llama_cpp/llama.py


class Assistant:
    def __init__(self):
        self.sentence_encoder = SentenceEncoder()
        self.index = Index().from_pretrained("resources/index_checkpoint")
        self.top_k = 10
        self.min_sim = 0.5
        self.max_len_context = 5000

    def __call__(self, question):  
        # Make sure the model path is correct for your system!
        llm = LlamaCpp(
            model_path="resources/llama-2-7b-chat.Q4_K_M.gguf",
            temperature=0.75,
            max_tokens=2000,
            top_p=1.,
            verbose=False, # Verbose is required to pass to the callback manager
            n_ctx=2048,
        )
        template =  """
                    Дополнительная информация, которая поможет ответить на вопрос: {info}
                    Контекст:
                    Вы услужливый, уважительный и честный ассистент банка Тинькофф. 
                    Всегда отвечайте как можно более полезно и безопасно. 
                    Ваши ответы не должны содержать никакого вредного, неэтичного, расистского, сексистского, токсичного, опасного или незаконного контента. 
                    Пожалуйста, убедитесь, что ваши ответы носят социально непредвзятый и позитивный характер. 
                    Если вопрос не имеет никакого смысла или фактически не согласован, объясните почему, вместо того чтобы отвечать на что-то неправильное. 
                    Если вы не знаете ответа на вопрос, пожалуйста, не распространяйте ложную информацию.
                    Отвечайте только на русском языке.
                    Пользователь: {question}
                    Ассистент:
                    """
        
        embedding = self.sentence_encoder.encode(question)
        sentences = self.index.find_top_k(embedding, top_k=self.top_k, min_sim=self.min_sim)
        info = ";".join(sentences)
        max_len = max(self.max_len_context - len(template), 0)
        info = info[:max_len]

        prompt = PromptTemplate(template=template, input_variables=["question", "info"])
        llm_chain = LLMChain(prompt=prompt, llm=llm)
        answer = llm_chain.run(question=question, info=info)
        return answer
    


