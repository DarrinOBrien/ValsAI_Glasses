"""Tinker sampling policy for the harness loop."""
import tinker
from tinker_cookbook import renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer

from cpp import tools

BASE_MODEL = "Qwen/Qwen3.5-9B"
RENDERER = "qwen3_5"


def get_renderer(model: str = BASE_MODEL, name: str = RENDERER):
    return renderers.get_renderer(name, get_tokenizer(model))


def system_messages(renderer) -> list[dict]:
    # Pass the full OpenAI tool objects ({"type":"function","function":{...}}), not tool_specs(): this is what
    # the HF chat template (and so vLLM) renders into the prompt. scripts.smoke_real verifies the match.
    return renderer.create_conversation_prefix_with_tools(tools.TOOLS, tools.SYSTEM_PROMPT)


def tinker_policy(sampling_client, renderer, max_tokens: int = 12000, temperature: float = 0.6,
                  top_p: float = 0.95, max_context: int = 60000):
    params = tinker.SamplingParams(max_tokens=max_tokens, temperature=temperature, top_p=top_p,
                                   stop=renderer.get_stop_sequences())

    async def policy(messages):
        prompt = renderer.build_generation_prompt(messages)
        if prompt.length > max_context:
            return {"role": "assistant", "content": "EXIT"}, True, 0
        res = await sampling_client.sample_async(prompt=prompt, num_samples=1, sampling_params=params)
        toks = res.sequences[0].tokens
        msg, term = renderer.parse_response(toks)
        return msg, bool(term.is_clean), len(toks)

    return policy


def make_sampling_client(model_path: str | None = None, base_model: str = BASE_MODEL):
    sc = tinker.ServiceClient()
    if model_path:
        return sc.create_sampling_client(model_path=model_path)
    return sc.create_sampling_client(base_model=base_model)


def hydrate(messages: list[dict]) -> list[dict]:
    """JSONL trajectories store tool calls as dicts; the renderer wants ToolCall objects."""
    out = []
    for m in messages:
        if m.get("tool_calls"):
            m = {**m, "tool_calls": [renderers.ToolCall(function=renderers.ToolCall.FunctionBody(
                name=t["function"]["name"], arguments=t["function"]["arguments"])) if isinstance(t, dict) else t
                for t in m["tool_calls"]]}
        out.append(m)
    return out
