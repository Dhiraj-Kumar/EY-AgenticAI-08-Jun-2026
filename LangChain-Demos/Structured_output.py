from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from typing import Literal
load_dotenv()

model = ChatOpenAI(model="gpt-4o-mini")

class Feedback(BaseModel):
    participant_name: str = Field(description="Name of the participant")
    summary: str = Field(description="Brief summary of the overall feedback")
    sentiment: Literal['positive', 'negative', 'neutral'] = Field(description="Sentiment if the feedback like positive, negative or neutral")
    highlights: list[str] = Field(description="List of positive highlights of the program described by the participant")
    lowlights: list[str] = Field(description="List of negative highlights of the program described by the participant")
    rating: int = Field(description="Rating of the program")
    email_address: str = Field(description="Email address of the participant")
    
structured_model = model.with_structured_output(Feedback)

response = structured_model.invoke("The Java Fullstack training program was well-structured and covered essential modules like Core Java, Spring Boot, Hibernate, and Angular. The hands-on projects and live coding sessions made it easier to apply concepts in real-world scenarios. The trainer was knowledgeable and supportive, and the sessions on Git and deployment provided a complete view of end-to-end development. However, the pace during the Spring Boot section felt a bit fast, and more time for practice would have been helpful. Additionally, a dedicated session on debugging and code optimization could enhance the learning experience. Some front-end sessions, especially on Angular, felt rushed, and could benefit from more real-time examples. Out of 5 I would give 4 rating for this program. Feedback given by Dhiraj Kumar. Email address - dhiraj2001@gmail.com")

print(response.sentiment)
print(response.rating)
print(response.summary)