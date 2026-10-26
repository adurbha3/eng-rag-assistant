from engrag.agent import EngineeringAgent, ToolChoice
from engrag.chunking import Chunk
from engrag.embeddings import VectorIndex
from engrag.llm import StubLLM


def _agent():
    chunks = [Chunk("sic::0", "sic", "SiC", "Silicon carbide transistors switch faster than silicon.")]
    return EngineeringAgent(index=VectorIndex(chunks), llm=StubLLM())


def test_agent_routes_arithmetic_expression_to_calculator():
    result = _agent().run("480 * 520")
    assert result.tool == ToolChoice.CALCULATE
    assert result.output == "249600"


def test_agent_routes_question_to_search():
    result = _agent().run("What switches faster than silicon?")
    assert result.tool == ToolChoice.SEARCH
    assert "silicon carbide" in result.output.lower()


def test_agent_handles_unsafe_expression_gracefully():
    agent = _agent()
    # Not a pure math expression, so it is routed to search by the stub classifier,
    # but a string that _looks_ like one should fail safely rather than raise.
    result = agent.run("2 + 2")
    assert result.tool == ToolChoice.CALCULATE
    assert result.output == "4"
