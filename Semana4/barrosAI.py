from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
import os
import shutil
from langchain_community.vectorstores import Chroma
import json
import requests
from langchain_core.embeddings import Embeddings
from dotenv import load_dotenv

load_dotenv()

PATH_TO_FILE = "rag_document.txt"
# load and read the file
with open(PATH_TO_FILE, "r", encoding="utf-8") as file:
    text = file.read()


# length-based splitting
splitter1 = RecursiveCharacterTextSplitter(
    separators=[""], chunk_size=800, chunk_overlap=100  # in characters  # in characters
)

# # text structure-based splitting
# splitter = RecursiveCharacterTextSplitter(
#     separators=["##", "---"],
#     chunk_size=128,  # in characters
#     chunk_overlap=64,  # in characters
# )

# markdown structure-based splitting
splitter2 = MarkdownHeaderTextSplitter(
    headers_to_split_on=[("#", "header-1"), ("##", "header-2"), ("*", "list")],
    strip_headers=False,  # keeps the headers
)

chunks1 = splitter1.create_documents([text])
chunks2 = splitter2.split_text(text)

embedder = HuggingFaceEmbeddings(
    model_name="nomic-ai/nomic-embed-text-v1.5",
    encode_kwargs={"normalize_embeddings": True},
)


# class ServerEmbeddings(Embeddings):
#     def __init__(self, server_url, api_key):
#         self.server_url = server_url
#         self.api_key = api_key

#     def _embed(self, texts):
#         payload = json.dumps({"model": "nomic-embed-text:latest", "input": texts})
#         response = requests.post(
#             self.server_url,
#             headers={
#                 "Authorization": f"Bearer {self.api_key}",
#                 "Content-Type": "application/json",
#             },
#             data=payload,
#         )
#         response.raise_for_status()
#         return response.json()["embeddings"]

#     def embed_documents(self, texts):
#         return self._embed(texts)

#     def embed_query(self, text):
#         return self._embed([text])[0]


# base_url = os.environ["OLLAMA_BASE_URL"]
# api_key = os.environ["OLLAMA_API_KEY"]
# url = base_url + "/api/embed"
# embedder = ServerEmbeddings(server_url=url, api_key=api_key)

CHROMA_PATH1 = "database1"
CHROMA_PATH2 = "database2"

# remove existing database
if os.path.exists(CHROMA_PATH1):
    shutil.rmtree(CHROMA_PATH1)
if os.path.exists(CHROMA_PATH2):
    shutil.rmtree(CHROMA_PATH2)

# create and persist the vector database
vectorstore1 = Chroma.from_documents(
    documents=chunks1,
    embedding=embedder,
    persist_directory=CHROMA_PATH1,
    collection_metadata={"hnsw:space": "cosine"},  # distance metric
)

vectorstore2 = Chroma.from_documents(
    documents=chunks2,
    embedding=embedder,
    persist_directory=CHROMA_PATH2,
    collection_metadata={"hnsw:space": "cosine"},  # distance metric
)

query = [
    "Qual é a idade de Inês Carvalho e em que bairro de Lisboa reside?",
    "Em que área académica concluiu a Inês o seu mestrado?",
    "Qual é o tópico principal da investigação de doutoramento da Inês?",
    "Qual é o cargo atual da Inês e em que setor de atividade trabalha?",
    "Que tecnologias e ferramentas a Inês utiliza diariamente no seu trabalho?",
    "Quantas vezes por semana a Inês costuma correr e ir ao ginásio?",
    "O que toma a Inês ao pequeno-almoço e qual é a sua posição em relação ao café?",
    "Como se chama o gato da Inês, qual a sua idade e qual é a sua cor?",
    "Que hábitos curiosos tem o gato Tiziano quando a Inês tenta trabalhar ou estudar?",
    "Quem é o pai da Inês e qual é a sua profissão/cargo desde 2024?",
    "Quem é a melhor amiga da Inês e a que se dedica profissionalmente?",
    "Quem é o vizinho do andar de cima da Inês e que tipo de estabelecimento possui?",
    "Quem é o dentista da Inês e em que cidade fica a sua clínica?",
    "Quem é a veterinária do gato Tiziano e onde está localizada a sua clínica?",
    "Qual é a profissão de Keanu Reeves e há quantos anos a exerce?",
    "Quem é o chef responsável por preparar a refeição favorita da família da Inês e qual é o prato?",
    "Qual foi a viagem favorita da Inês e que características procura num destino?",
    "Quais são os países ou regiões que a Inês mais gostaria de visitar no futuro?",
    "Que tipos de filmes a Inês prefere e quais é que evita assistir?",
    "Que línguas a Inês fala fluentemente e qual pretende aprender nos próximos anos?",
]

retriever1 = vectorstore1.as_retriever(search_kwargs={"k": 3})
retriever2 = vectorstore2.as_retriever(search_kwargs={"k": 3})

# retrieve most relevant chunks (processa a lista completa de perguntas)
retrieved_docs1 = retriever1.batch(query)
retrieved_docs2 = retriever2.batch(query)

# retrieve chunks together with distance scores
results_with_scores1 = vectorstore1.similarity_search_with_score(query[0], k=3)
results_with_scores2 = vectorstore2.similarity_search_with_score(query[0], k=3)

print(results_with_scores1)
