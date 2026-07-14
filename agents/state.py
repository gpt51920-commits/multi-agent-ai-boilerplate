from typing import Annotated, Literal
from pydantic import BaseModel, Field
from langgraph.graph.message import add_messages

class AgentState(BaseModel):
    """Стан графа — єдине джерело правди між нодами."""
    messages: Annotated[list, add_messages]
    task: str = ""
    research: str = ""
    draft: str = ""
    feedback: str = ""
    status: Literal["pending", "researching", "writing", "reviewing", "done", "failed"] = "pending"
    iteration: int = 0
    max_iterations: int = 3