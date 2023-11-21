FROM python:3.9.16

EXPOSE 8080

WORKDIR /code

RUN pip install --upgrade pip
RUN pip install torch, langchain, llama-cpp-python, sentence-transformers, uvicorn

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]