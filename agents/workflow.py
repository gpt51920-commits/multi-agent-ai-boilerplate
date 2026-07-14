from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from app.agents.state import AgentState
from app.agents.nodes import researcher, writer, reviewer, route_after_review

def build_graph():
    # Створюємо граф на основі нашого стану
    graph = StateGraph(AgentState)
    
    # Додаємо вузли (агентів)
    graph.add_node("researcher", researcher)
    graph.add_node("writer", writer)
    graph.add_node("reviewer", reviewer)

    # Визначаємо порядок виконання
    graph.add_edge(START, "researcher")
    graph.add_edge("researcher", "writer")
    graph.add_edge("writer", "reviewer")
    
    # Умовний перехід: якщо редактор не задоволений, повертаємось до письменника
    graph.add_conditional_edges("reviewer", route_after_review, {"writer": "writer", "end": END})

    # Додаємо пам'ять для збереження історії
    memory = MemorySaver()
    return graph.compile(checkpointer=memory)