import os

from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END, MessagesState


load_dotenv()

base_url = os.environ["OLLAMA_BASE_URL"]
api_key = os.environ["OLLAMA_API_KEY"]
model = os.environ["OLLAMA_MODEL"]

llm = ChatOllama(
    model=model,
    base_url=base_url,
    client_kwargs={
        "headers": {
            "Authorization": f"Bearer {api_key}"
        }
    }
)


# Nó "model": recebe o estado e chama o LLM
def call_model(state: MessagesState):
    response = llm.invoke(state["messages"])
    return {"messages": [response]}


# Workflow: START -> model -> END
builder = StateGraph(MessagesState)

builder.add_node("model", call_model)

builder.add_edge(START, "model")
builder.add_edge("model", END)

graph = builder.compile()


# Visualização do workflow
print(graph.get_graph().draw_mermaid())

with open("workflow.png", "wb") as f:
    f.write(graph.get_graph().draw_mermaid_png())


# Chat
messages = []

while True:
    prompt = input("TU: ")

    if prompt == "sair":
        break

    messages.append(("user", prompt))

    result = graph.invoke({
        "messages": messages
    })

    messages = result["messages"]

    print(messages)
    print("IA:", messages[-1].content)