from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langgraph.types import Command
from langgraph.checkpoint.memory import InMemorySaver
from dotenv import load_dotenv
from langchain.tools import tool
from typing import Literal
import os
import resend
import streamlit as st

st.header("Email Sender Agent")

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini")

class Feedback(BaseModel):
    participant_name: str = Field(description="Name of the participant")
    summary: str = Field(description="Brief summary of the overall feedback")
    sentiment: Literal['positive', 'negative', 'neutral'] = Field(description="Sentiment if the feedback like positive, negative or neutral")
    highlights: list[str] = Field(description="List of positive highlights of the program described by the participant")
    lowlights: list[str] = Field(description="List of negative highlights of the program described by the participant")
    rating: int = Field(description="Rating of the program")
    email_address: str = Field(description="Email address of the participant")

@tool
def send_email(email_address: str, body: str):
    """
    Tool for sending email
    """
    resend.api_key=os.getenv("RESEND_KEY")
    params: resend.Emails.SendParams = {
        "from": "training@resend.dev",
        "to": [email_address],
        "subject": "Reply from Training Manager",
        "html": body
    }
    email = resend.Emails.send(params)
    return email


agent = create_agent(
    model=llm,
    response_format=Feedback,
    tools=[send_email],
    system_prompt="You are an AI assistant who analyze the customer feedback and send an email to customer based on feedback sentiment. If sentiment is positive, you send a Thank you email and if sentiment is negative you send an Apology email. You use tool to send email. Use html format for drafting an email. You can use emojis but keep the tone professional.",
    checkpointer=InMemorySaver(),
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={
                "send_email":{
                    "allowed_decisions": ["approve", "reject"]
                }
            }
        )
    ]
)
feedback_input = st.text_area("Enter your feedback")
if st.button("Start Workflow"):
    result = agent.invoke({
        "messages": [
            {"role": "user", "content": feedback_input}
        ]
    }, config={"configurable": {"thread_id": "1"}})

    if "__interrupt__" in result:
        st.write("Workflow paused... Waiting for human approval")
        st.write("1. Approve\n2. Reject")
        user_input=st.text_input("Enter your choice")
        choices = {"1": "approve", "2": "reject"}
        if st.button("Submit"):
            st.write("Resuming....")
            result = agent.invoke(
                Command(
                    resume={
                        "decisions": [
                            {
                                "type": choices.get(user_input)
                            }
                        ]
                    }
                ), config={"configurable": {"thread_id": "1"}}
            )
        
            st.write("Workflow completed")
    # print(result['messages'][-1].content)