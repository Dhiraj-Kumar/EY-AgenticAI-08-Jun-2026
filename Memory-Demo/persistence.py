from langchain_openai import ChatOpenAI
from langgraph.checkpoint.postgres import PostgresSaver
from langchain.agents.middleware import SummarizationMiddleware
from langchain.agents import create_agent
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini")

DB_URI="postgresql://postgres:niit1234@localhost:5432/conversationdb?sslmode=disable"

with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
    checkpointer.setup()
    agent = create_agent(
        model = llm,
        system_prompt="You are a helpful AI assistant who answers query asked by the user with polite and professional tone.",
        checkpointer=checkpointer,
        middleware=[
            SummarizationMiddleware(
                # trigger=("tokens", 4000)
                trigger=("fraction", 0.2),
                keep=("messages", 5)
            )
        ]
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