import os
import json
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langfuse.langchain import CallbackHandler
from pydantic import BaseModel
from typing import Literal

# Groq API Key
os.environ["GROQ_API_KEY"] = "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"

# LangFuse API Keys (ТЕПЕР ЧЕРЕЗ os.environ!)
os.environ["LANGFUSE_PUBLIC_KEY"] = "pk-lf-b506bca9-e27f-4831-a806-4a46ce5ac3ce"
os.environ["LANGFUSE_SECRET_KEY"] = "sk-lf-d23d453d-9aea-41ec-a2aa-eae7be7911a7"
os.environ["LANGFUSE_HOST"] = "https://cloud.langfuse.com"

# Ініціалізація LangFuse callback (БЕЗ аргументів!)
langfuse_handler = CallbackHandler()

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.2,
    groq_api_key=os.environ["GROQ_API_KEY"]
)

class SupportState(BaseModel):
    user_query: str = ""
    category: Literal["billing", "technical", "general", "urgent"] = "general"
    priority: Literal["low", "medium", "high"] = "medium"
    response_draft: str = ""
    escalated: bool = False
    knowledge_context: str = ""

async def classify_intent(state: SupportState) -> dict:
    prompt = f"""Класифікуй запит підтримки. Поверни ТІЛЬКИ JSON без додаткового тексту:
{{"category": "billing|technical|general|urgent", "priority": "low|medium|high"}}

Запит користувача: {state.user_query}"""
    
    resp = await llm.ainvoke(
        [
            SystemMessage(content="Ти класифікатор запитів підтримки. Повертай тільки JSON."),
            HumanMessage(content=prompt),
        ],
        config={"callbacks": [langfuse_handler]}
    )
    
    try:
        content = resp.content.strip()
        if content.startswith("```json"):
            content = content.replace("```json", "").replace("```", "").strip()
        result = json.loads(content)
        return {
            "category": result.get("category", "general"),
            "priority": result.get("priority", "medium")
        }
    except Exception as e:
        print(f"Error parsing JSON: {e}")
        return {"category": "general", "priority": "medium"}

async def retrieve_knowledge(state: SupportState) -> dict:
    fake_context = """Base URL: example.com
API docs: /api/v1/docs
Billing email: billing@example.com
Technical support: support@example.com
Refund policy: 30 days money-back guarantee"""
    return {"knowledge_context": fake_context}

async def generate_response(state: SupportState) -> dict:
    prompt = f"""Ти агент підтримки. Відповідай чітко, професійно та дружелюбно.
Використовуй наданий контекст.

Контекст:
{state.knowledge_context}

Запит користувача: {state.user_query}
Категорія: {state.category}
Пріоритет: {state.priority}

Напиши відповідь:"""
    
    resp = await llm.ainvoke(
        [
            SystemMessage(content="Ти агент підтримки. Відповідай чітко та професійно."),
            HumanMessage(content=prompt),
        ],
        config={"callbacks": [langfuse_handler]}
    )
    return {"response_draft": resp.content}

async def route_to_human(state: SupportState) -> dict:
    if state.priority == "high" or state.category == "urgent":
        return {"escalated": True}
    return {"escalated": False}

def build_support_graph():
    graph = StateGraph(SupportState)
    graph.add_node("classify", classify_intent)
    graph.add_node("retrieve", retrieve_knowledge)
    graph.add_node("generate", generate_response)
    graph.add_node("route", route_to_human)

    graph.add_edge(START, "classify")
    graph.add_edge("classify", "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", "route")
    graph.add_edge("route", END)

    memory = MemorySaver()
    return graph.compile(checkpointer=memory)