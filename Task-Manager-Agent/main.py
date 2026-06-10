from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.tools import tool
from dotenv import load_dotenv
import sqlite3

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini")

con = sqlite3.connect("tasks.db", check_same_thread=False)
cursor = con.cursor()

# Create table if not exists
cursor.execute("""
create table if not exists todos (
    id integer primary key autoincrement,
    todo text not null,
    iscompleted boolean not null
)
""")
con.commit()

@tool
def add_todo(todo: str):
    """
    Add a new todo in todos table in database
    """
    cursor.execute("insert into todos (todo, iscompleted) values (?,?)",(todo, False))
    con.commit()

@tool
def get_all_todos():
    """
    Get all todos from todo table
    """
    cursor.execute("Select id, todo, iscompleted from todos order by id")
    return cursor.fetchall()
    
@tool
def get_todo_by_id(todo_id: str):
    """
    Get todo item from todos table based on todo_id
    """
    cursor.execute("Select id, todo, iscompleted from todos where id=?",(todo_id))
    return cursor.fetchall()

@tool 
def delete_todo(todo_id: str):
    """
    Delete a todo from todos table based on todo_id
    """
    cursor.execute("Delete from todos where id=?", (todo_id))
    con.commit()

@tool
def update_todo(todo_id: str, completed=True):
    """
    Update the status of a todo item to True or False in todos table based on todo task completion
    """
    cursor.execute("Update todos set iscompleted=? where id=?", (completed, todo_id))
    con.commit()
    

agent = create_agent(
    model = llm,
    system_prompt="You are an AI task manager who helps in managing tasks for the user. You call tools to perform different actions. You are strictly managing the todos and do not answer queries related to anything else",
    tools=[add_todo, get_all_todos, get_todo_by_id, delete_todo, update_todo]
)

while True:
    user_input = input("You: ")
    if user_input.lower()=="exit":
        break
    response = agent.invoke({
        "messages": [
            {"role": "user", "content": user_input}
        ]
    })
    print(f"AI: {response['messages'][-1].content}")