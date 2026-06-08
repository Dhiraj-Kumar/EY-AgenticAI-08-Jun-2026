from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()

model = ChatOpenAI(model="gpt-4o-mini")

chat_history=[]

while True:
    user_input = input("You: ")
    chat_history.append(user_input)
    if user_input.lower() == "exit":
        break
    response = model.invoke(chat_history)
    chat_history.append(response.content)
    print(f"AI: {response.content}")

print(chat_history)