from langgraph.graph import StateGraph, END
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate

# 初始化 LLM
llm = ChatOpenAI(model="gpt-xxx", temperature=0.7)

# 定义状态结构
state = {
    "task": "Find the sum of all even numbers from 1 to 10.",
    "memory": [],
    "attempt": 0,
    "output": None,
    "feedback": None,
    "reflection": None,
}

# 1️⃣ 行动节点：模型尝试解题
def act(state):
    prompt = f"""
You are solving the task: {state['task']}

Past reflections:
{state['memory']}

Think step-by-step and output your final answer clearly.
"""
    result = llm.invoke(prompt).content
    state["output"] = result
    return state

# 2️⃣ 评估节点：检查是否正确
def evaluate(state):
    expected_answer = "30"
    if expected_answer in state["output"]:
        state["feedback"] = "✅ Correct!"
    else:
        state["feedback"] = f"❌ Incorrect. Expected {expected_answer}, got {state['output'][:50]}"
    return state

# 3️⃣ 反思节点：总结错误经验
def reflect(state):
    if "✅" in state["feedback"]:
        return state  # 无需反思
    reflection_prompt = f"""
You received the feedback: {state['feedback']}
Based on your previous reflections: {state['memory']}
Write one sentence describing what went wrong and how to avoid it next time.
"""
    reflection = llm.invoke(reflection_prompt).content
    state["reflection"] = reflection
    state["memory"].append(reflection)
    return state

# 4️⃣ 控制节点：决定是否继续
def decide(state):
    if "✅" in state["feedback"] or len(state["memory"]) > 3:
        return END
    return "act"

# 构建 Reflexion Agent 图
graph = StateGraph()
graph.add_node("act", act)
graph.add_node("evaluate", evaluate)
graph.add_node("reflect", reflect)

graph.add_edge("act", "evaluate")
graph.add_edge("evaluate", "reflect")
graph.add_conditional_edges("reflect", decide)

app = graph.compile()

# 运行 Reflexion Agent
current_state = state
for i in range(5):
    current_state["attempt"] += 1
    current_state = app.invoke(current_state)
    print(f"\n===== Attempt {i+1} =====")
    print("Output:", current_state["output"])
    print("Feedback:", current_state["feedback"])
    print("Reflection:", current_state.get("reflection"))
    if "✅" in current_state["feedback"]:
        break

print("\n🧠 Final Memory of Reflections:")
for i, m in enumerate(current_state["memory"], 1):
    print(f"{i}. {m}")
