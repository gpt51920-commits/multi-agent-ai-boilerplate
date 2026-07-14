import os
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from langfuse.langchain import CallbackHandler
from app.agents.state import AgentState

# Groq API Key
os.environ["GROQ_API_KEY"] = "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"

# LangFuse API Keys (ТЕПЕР ЧЕРЕЗ os.environ!)
os.environ["LANGFUSE_PUBLIC_KEY"] = "pk-lf-b506bca9-e27f-4831-a806-4a46ce5ac3ce"
os.environ["LANGFUSE_SECRET_KEY"] = "sk-lf-d23d453d-9aea-41ec-a2aa-eae7be7911a7"
os.environ["LANGFUSE_HOST"] = "https://cloud.langfuse.com"

# Ініціалізація LangFuse callback (БЕЗ аргументів!)
langfuse_handler = CallbackHandler()

# Ініціалізація моделі Llama 3 через Groq
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.3,
    groq_api_key=os.environ["GROQ_API_KEY"]
)

async def researcher(state: AgentState) -> dict:
    resp = await llm.ainvoke(
        [
            SystemMessage(content="Ти дослідник. Збирай ключові факти."),
            HumanMessage(content=state.task),
        ],
        config={"callbacks": [langfuse_handler]}
    )
    return {
        "research": resp.content,
        "status": "writing",
        "messages": state.messages + [HumanMessage(content=resp.content)]
    }

async def writer(state: AgentState) -> dict:
    resp = await llm.ainvoke(
        [
            SystemMessage(content="Ти письменник. Напиши чіткий текст на основі дослідження."),
            HumanMessage(content=f"Дослідження:\n{state.research}\n\nЗадача: {state.task}"),
        ],
        config={"callbacks": [langfuse_handler]}
    )
    return {"draft": resp.content, "status": "reviewing"}

async def reviewer(state: AgentState) -> dict:
    resp = await llm.ainvoke(
        [
            SystemMessage(content="Ти редактор. Оціни текст за 10-бальною шкалою. Якщо <8 — дай конкретні правки."),
            HumanMessage(content=f"Текст:\n{state.draft}\n\nЗадача: {state.task}"),
        ],
        config={"callbacks": [langfuse_handler]}
    )
    new_iter = state.iteration + 1
    if "8" in resp.content or "9" in resp.content or "10" in resp.content or new_iter >= state.max_iterations:
        return {"feedback": resp.content, "status": "done", "iteration": new_iter}
    return {"feedback": resp.content, "status": "writing", "iteration": new_iter}

def route_after_review(state: AgentState) -> str:
    return "end" if state.status == "done" else "writer"