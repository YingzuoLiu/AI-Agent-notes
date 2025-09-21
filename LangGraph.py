from langgraph.graph import StateGraph
import random
import matplotlib.pyplot as plt

# 定义状态
class RecState(dict):
    pass

# --- 节点定义 ---

def router_node(state):
    """决定探索还是利用"""
    eps = state["epsilon"]
    if random.random() < eps:
        state["path"] = "explore"
    else:
        state["path"] = "rank"
    return state

def explore_node(state):
    """探索阶段，随机推荐"""
    state["action"] = random.choice(range(state["n_items"]))
    state["reward"] = random.choice([0, 1])  # 模拟用户反馈
    return state

def ranking_node(state):
    """利用阶段，选择最高分物品"""
    state["action"] = 0  # 简化：总选第0个
    state["reward"] = 1  # 假设选对了有正反馈
    return state

def update_node(state):
    """更新参数，记录历史"""
    # 记录历史
    state["history"].append({
        "step": state["step"],
        "epsilon": state["epsilon"],
        "reward": state["reward"]
    })
    # 衰减 epsilon
    state["epsilon"] *= state["epsilon_decay"]
    state["step"] += 1
    return state

# --- 构建 LangGraph ---
def build_graph():
    graph = StateGraph(RecState)

    # 1. 添加节点
    graph.add_node("router", router_node)
    graph.add_node("explore", explore_node)
    graph.add_node("rank", ranking_node)
    graph.add_node("update", update_node)

    # 2. 设置入口节点
    graph.set_entry_point("router")

    # 3. 设置 router 的分支路径
    graph.add_conditional_edges(
        "router",                # 当前节点
        lambda s: s["path"],     # 决策函数，返回 "explore" 或 "rank"
        {
            "explore": "explore",  # 如果 path=="explore"，就跳到 explore 节点
            "rank": "rank"         # 如果 path=="rank"，就跳到 rank 节点
        }
    )

    # 4. 画出后续的顺序边
    graph.add_edge("explore", "update")
    graph.add_edge("rank", "update")
    graph.add_edge("update", "router")

    return graph.compile()


# --- 运行实验 ---
def run_experiment(epsilon=0.5, epsilon_decay=0.95, steps=50):
    state = RecState(
        step=0,
        n_items=10,
        epsilon=epsilon,
        epsilon_decay=epsilon_decay,
        history=[]
    )
    app = build_graph()
    for _ in range(steps):
        state = app.invoke(state)
    return state

# --- 主程序 ---
if __name__ == "__main__":
    final_state = run_experiment(epsilon=0.8, epsilon_decay=0.97, steps=50)

    # 可视化
    epsilons = [h["epsilon"] for h in final_state["history"]]
    rewards = [h["reward"] for h in final_state["history"]]

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epsilons)
    plt.title("Epsilon Decay")
    plt.xlabel("Step")
    plt.ylabel("Epsilon")

    plt.subplot(1, 2, 2)
    plt.plot(rewards)
    plt.title("Rewards over Time")
    plt.xlabel("Step")
    plt.ylabel("Reward")
    plt.tight_layout()
    plt.show()
