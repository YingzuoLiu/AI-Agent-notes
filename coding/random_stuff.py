def merge_topk(results_list, k):
    heap = []
    final_plan = []
    seen = set()

    for sid, doc in enumerate(results_list):
        if doc:
            option = doc[0]
            heapq.heappush(
                heap,(-option["score"],sid,0,option)
            )
    while heap and len(final_plan) < k:
        neg_score,sid,idx,option = heapq.heappop(heap)
        plan_id = option["id"]
        if plan_id in seen:
            continue

        final_plan.append(option["doc"])
        seen.add(plan_id)

        next_idx = idx + 1
        if next_idx < len(results_list[sid]):
            next_option = results_list[sid][next_idx]
            heapq.heappush(
                heap,
                (-next_option["score"],sid,next_idx,next_option)
            )

    return final_plan

def call_with_retry(tool_fn,query,retry_limit=2):
    for _ in range(retry_limit):
        try:
            result = tool_fn(query)
            if result:
                return result
        expect Expection:
            continue
    return "fallback"


def chunk_text(text,size=300,overlap=50):
    chunks = []
    step = size - overlap

    for i in range(0,len(text),step):
        chunk = text[i:i+size]
        chunks.append(chunk)

        if i+size >= len(text):
            break

    return chunks

def route_query(query):
    for keyword,tool in tool_registry.items():
        if keyword in query:
            return tool
    return "faq_tool"
