class WealthRecommendationSkill:
    def __init__(self, retriever, ranker, validator, llm):
        self.retriever = retriever
        self.ranker = ranker
        self.validator = validator
        self.llm = llm

    def run(self, state):
        # 1) retrieval
        candidates = self.retriever.search(
            query=state["risk_level"],
            top_k=20
        )

        # 2) ranking
        ranked = self.ranker.rank(
            query=state["user_query"],
            candidates=candidates
        )

        # 3) business rule validation
        valid_products = self.validator.filter(
            ranked,
            risk_level=state["risk_level"],
            duration=state["duration"]
        )

        # 4) explanation generation
        prompt = f"""
        用户需求: {state['user_query']}
        候选产品: {valid_products[:3]}
        请生成推荐解释
        """
        explanation = self.llm.generate(prompt)

        return {
            "products": valid_products[:3],
            "reason": explanation
        }
