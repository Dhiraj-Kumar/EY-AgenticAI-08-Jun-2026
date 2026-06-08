from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from dotenv import load_dotenv
load_dotenv()

model = ChatOpenAI(model="gpt-4o-mini")

chat_history=[
    SystemMessage(content="You are a helpful AI assistant who answers query with a bit of humour. Use emojis while generating responses. Maintain a professional tone.")
]

while True:
    user_input = input("You: ")
    chat_history.append(HumanMessage(content=user_input))
    if user_input.lower() == "exit":
        break
    response = model.invoke(chat_history)
    chat_history.append(AIMessage(content=response.content))
    print(f"AI: {response.content}")

print(chat_history)