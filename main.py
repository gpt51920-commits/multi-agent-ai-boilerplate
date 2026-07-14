import os
import uuid
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from app.agents.workflow import build_graph
from app.agents.support_workflow import build_support_graph

app = FastAPI(title="Multi-Agent Boilerplate")

class TaskRequest(BaseModel):
    task: str

class SupportRequest(BaseModel):
    query: str

@app.get("/")
def read_root():
    return {
        "status": "ok", 
        "message": "Multi-Agent Boilerplate з Groq успішно запущено!",
        "endpoints": {
            "/run": "Research → Write → Review Agent",
            "/support": "Customer Support Router",
            "/docs": "Swagger UI"
        }
    }

@app.post("/run")
async def run_agent(request: TaskRequest):
    try:
        app_graph = build_graph()
        initial_state = {
            "messages": [],
            "task": request.task,
            "research": "",
            "draft": "",
            "feedback": "",
            "status": "pending",
            "iteration": 0,
            "max_iterations": 3
        }
        config = {"configurable": {"thread_id": str(uuid.uuid4())}}
        final_state = await app_graph.ainvoke(initial_state, config)
        
        return {
            "status": final_state["status"],
            "draft": final_state.get("draft"),
            "research": final_state.get("research"),
            "iterations": final_state.get("iteration", 0)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/support")
async def run_support(request: SupportRequest):
    """Ендпоінт для Customer Support Router."""
    try:
        support_graph = build_support_graph()
        initial_state = {
            "user_query": request.query,
            "category": "general",
            "priority": "medium",
            "response_draft": "",
            "escalated": False,
            "knowledge_context": ""
        }
        config = {"configurable": {"thread_id": str(uuid.uuid4())}}
        final_state = await support_graph.ainvoke(initial_state, config)
        
        return {
            "thread_id": config["configurable"]["thread_id"],
            "category": final_state["category"],
            "priority": final_state["priority"],
            "response": final_state["response_draft"],
            "escalated": final_state["escalated"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))