from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents import create_agent
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini")

agent = create_agent(
    model = llm,
    system_prompt="You are a helpful AI assistant who answers query asked by the user with polite and professional tone.",
    checkpointer=InMemorySaver()
)

while True:
    user_input = input("You: ")
    if user_input.lower()=="exit":
        break
    result = agent.invoke({
        "messages": [
            {"role": "user", "content": user_input}
        ]
    }, config={"configurable": {"thread_id": "1"}})
    print(result['messages'][-1].content)