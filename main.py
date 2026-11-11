
from strands import Agent, tool
from bedrock_agentcore.runtime import BedrockAgentCoreApp
from bedrock_agentcore.memory import MemoryClient
from strands.models import BedrockModel
from strands.tools.mcp.mcp_client import MCPClient
from mcp.client.streamable_http import streamable_http_client
import argparse, json
import os, asyncio, boto3
from strands.hooks import (
    HookProvider, AfterInvocationEvent, HookRegistry, MessageAddedEvent,
)
import logging
import uuid
from typing import Dict
from bedrock_agentcore.tools.code_interpreter_client import code_session
from strands_tools.browser import AgentCoreBrowser
from decimal import Decimal, ROUND_HALF_UP


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("CSAI_Agent")
app = BedrockAgentCoreApp()

# Suppress interactive tool-consent prompts (required in headless deployments).
os.environ["BYPASS_TOOL_CONSENT"] = "true"

GATEWAY_URL = "https://customersupportgateway-scgtsommoe.gateway.bedrock-agentcore.us-east-1.amazonaws.com/mcp"
KB_ID       = "FCJRXKM1SQ"
REGION      = "us-east-1"
MEMORY_ID   = "CustomerSupportMemory-nLyb86B8gC"

model_id = "global.amazon.nova-2-lite-v1:0"

model = BedrockModel(
    model_id=model_id,
    region_name=REGION,
    max_tokens=4096,
)
memory_client = MemoryClient(region_name=REGION)

_bedrock_runtime = boto3.client(
    "bedrock-agent-runtime",
    region_name=REGION,
)

def get_namespaces(mem_client: MemoryClient, memory_id: str) -> Dict:
    strategies = mem_client.get_memory_strategies(memory_id)

    namespaces = {}
    for strategy in strategies:
        templates = (
            strategy.get("namespaceTemplates")
            or strategy.get("namespaces")
            or []
        )

        if templates:
            namespaces[strategy["type"]] = templates[0]

    return namespaces

class MemoryHook(HookProvider):
    """Long-term memory hook for the customer support agent."""

    def __init__ (self, actor_id, session_id, memory_client, memory_id):
        self.actor_id = actor_id
        self.session_id = session_id
        self.memory_client = memory_client
        self.memory_id = memory_id
        self.namespaces = get_namespaces(self.memory_client,self.memory_id,)

    def register_hooks(self, registry: HookRegistry) -> None:
        registry.add_callback(MessageAddedEvent,self.retrieve_customer_context)
        registry.add_callback(AfterInvocationEvent,self.save_support_interaction)
    def retrieve_customer_context(self, event: MessageAddedEvent):
        """Retrieve relevant memories and prepend them to the user message."""
        messages = event.agent.messages
        if not messages:
            return
        message = messages[-1]
        content = message.get("content", [])
        if (message.get("role") != "user" or not content or any("toolResult" in block for block in content)):
            return


        text_block = next((block for block in content if "text" in block),None)
        if text_block is None:
            return

        query = text_block["text"]
        self.latest_customer_query = query
        try:
            context_items = []

            for strategy_type, template in self.namespaces.items():
                namespace = template.format(actorId=self.actor_id)

                memories = self.memory_client.retrieve_memories(
                    memory_id=self.memory_id,
                    namespace=namespace,
                    query=query,
                    top_k=5,
                )

                for memory in memories:
                    text = memory.get("content", {}).get("text", "").strip()
                    if text:
                        context_items.append(
                            f"[{strategy_type}] {text}"
                        )

            if context_items:
                text_block["text"] = ("Customer Context:\n"+ "\n".join(context_items)+ "\n\n"+ query)

        except Exception:
            logger.exception("Failed to retrieve customer memories")

    def save_support_interaction(self, event: AfterInvocationEvent):
        try:
            customer_query = None
            agent_response = None

            for message in reversed(event.agent.messages):
                content = message.get("content", [])

                # Tool results are not customer questions.
                if any("toolResult" in block for block in content):
                    continue

                text = "\n".join(
                    block["text"]
                    for block in content
                    if "text" in block
                ).strip()

                if not text:
                    continue

                if message["role"] == "assistant" and agent_response is None:
                    agent_response = text

                elif message["role"] == "user" and agent_response is not None:
                    customer_query = getattr(
                        self,
                        "latest_customer_query",
                        text,
                    )
                    break

            if customer_query and agent_response:
                self.memory_client.create_event(
                    memory_id=self.memory_id,
                    actor_id=self.actor_id,
                    session_id=self.session_id,
                    messages=[
                        (customer_query, "USER"),
                        (agent_response, "ASSISTANT"),
                    ],
                )

        except Exception:
            logger.exception("Failed to save support interaction")

@tool
def search_knowledge_base(query: str) -> str:
    """Search product specifications, return policies, warranties,
    and loyalty program information.

    Args:
        query: The question or topic to search for.

    Returns:
        Relevant text retrieved from the knowledge base.
    """
    if not KB_ID:
        return "Knowledge base not configured."

    try:
        response = _bedrock_runtime.retrieve(
            knowledgeBaseId=KB_ID,
            retrievalQuery={"text": query},
        )

        results = response.get("retrievalResults", [])

        chunks = [
            result.get("content", {}).get("text", "").strip()
            for result in results
        ]
        chunks = [text for text in chunks if text]

        if not chunks:
            return "No relevant information found in the knowledge base."

        return "\n---\n".join(chunks)

    except Exception:
        logger.exception("Knowledge Base retrieval failed")
        return "Knowledge Base retrieval failed. Do not guess the answer."


@tool
def calculate_loyalty_discount(loyalty_points: int,tier: str,order_total: float,product_category: str = "standard",) -> str:
    """Calculate points redemption, tier discount, final total,
    earned points, and remaining points using a Python sandbox.

    Args:
        loyalty_points: Current customer points balance.
        tier: Silver, Gold, or Platinum.
        order_total: Order amount in USD.
        product_category: standard, device, or fresh.
    """
    tier = tier.strip().title()
    product_category = product_category.strip().lower()

    if loyalty_points < 0 or order_total < 0:
        return "Points and order total must not be negative."

    if tier not in {"Silver", "Gold", "Platinum"}:
        return "Tier must be Silver, Gold, or Platinum."

    if product_category not in {"standard", "device", "fresh"}:
        return "Category must be standard, device, or fresh."

    code = f"""
import json
from decimal import Decimal, ROUND_HALF_UP

points = {loyalty_points!r}
tier = {tier!r}
category = {product_category!r}
order_total = Decimal({str(order_total)!r})

earn_rates = {{"standard": 1, "device": 2, "fresh": 5}}
tier_rates = {{
    "Silver": Decimal("0.00"),
    "Gold": Decimal("0.10"),
    "Platinum": Decimal("0.15"),
}}

def money(value):
    return value.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

available_points = (points // 500) * 500
maximum_points = int(order_total * Decimal("0.50") * 100)
maximum_points = (maximum_points // 500) * 500

points_redeemed = min(available_points, maximum_points)
points_discount = Decimal(points_redeemed) / 100
subtotal = order_total - points_discount

tier_discount = money(subtotal * tier_rates[tier])
final_total = money(subtotal - tier_discount)
total_savings = money(order_total - final_total)

points_earned = int(final_total * earn_rates[category])
remaining_points = points - points_redeemed + points_earned

print(json.dumps({{
    "order_total": float(money(order_total)),
    "points_redeemed": points_redeemed,
    "points_discount": float(points_discount),
    "tier": tier,
    "tier_discount": float(tier_discount),
    "final_total": float(final_total),
    "total_savings": float(total_savings),
    "points_earned": points_earned,
    "remaining_points": remaining_points,
    "tier_discount_pct": float(tier_rates[tier]),
}}))
"""

    try:
        with code_session(REGION) as code_client:
            response = code_client.invoke(
                "executeCode",
                {
                    "code": code,
                    "language": "python",
                    "clearContext": True,
                },
            )

            for event in response["stream"]:
                if "result" in event:
                    result = event["result"]
                    if result.get("isError"):
                        raise RuntimeError(str(result))
                    return json.dumps(result)

        raise RuntimeError("Code Interpreter returned no result")

    except Exception:
        logger.exception(
            "Code Interpreter failed; using tier-only fallback"
        )

        rates = {
            "Silver": Decimal("0.00"),
            "Gold": Decimal("0.10"),
            "Platinum": Decimal("0.15"),
        }

        total = Decimal(str(order_total))
        discount = (total * rates[tier]).quantize(
            Decimal("0.01"),
            rounding=ROUND_HALF_UP,
        )

        return json.dumps({
            "fallback": True,
            "message": (
                "Code Interpreter unavailable. Applied tier discount "
                "only; points redemption and earning were not calculated."
            ),
            "points_redeemed": 0,
            "tier_discount_pct": float(rates[tier]),
            "tier_discount": float(discount),
            "final_total": float(total - discount),
            "remaining_points": loyalty_points,
        })
SYSTEM_PROMPT = """You are a customer support assistant for an online store.

Use your tools to:
- Look up customers, orders, shipping details, refunds, and return labels.
- Search the Knowledge Base for product information and store policies.
- Calculate loyalty discounts using calculate_loyalty_discount.
- Browse websites when the customer requests live information.

Use retrieved Customer Context to personalise your responses.
Do not invent customer details, tool results, or store policies.

For refund requests, look up the order first to obtain its details and amount.
For calculations, use the calculation tool and clearly disclose any fallback.
If a tool fails, explain the failure rather than claiming success.

Keep responses concise unless the customer asks for more detail.
"""
@app.entrypoint
async def invoke(payload, context=None):
    """Handle a customer support request."""
    user_input = payload.get("prompt", "").strip()

    if not user_input:
        return "Please provide a question in the prompt field."

    actor_id = payload.get("customer_id", "anonymous")

    session_id = (
        payload.get("session_id")
        or getattr(context, "session_id", None)
        or str(uuid.uuid4())
    )

    try:
        memory_hook = MemoryHook(
            actor_id=actor_id,
            session_id=session_id,
            memory_client=memory_client,
            memory_id=MEMORY_ID,
        )

        agent_core_browser = AgentCoreBrowser(
            region=REGION,
            session_timeout=600,
        )

        tools = [
            search_knowledge_base,
            calculate_loyalty_discount,
            agent_core_browser.browser,
        ]

        gateway_client  = MCPClient(
            lambda: streamable_http_client(url=GATEWAY_URL)
        )

        with gateway_client :
            try:
                gateway_tools = gateway_client.list_tools_sync()
                tools.extend(gateway_tools)

                logger.info(
                    "Gateway connected successfully. Loaded %d tools.",
                    len(gateway_tools),
                )

            except TimeoutError:
                logger.exception("Gateway tool loading timed out")
                return "The support tools timed out. Please try again shortly."

            except ConnectionError:
                logger.exception("Gateway connection failed")
                return "The support tools could not be reached. Please try again shortly."

            except Exception:
                logger.exception("Gateway tool loading failed")
                return (
                    "The support tools are unavailable. "
                    "Please try again later."
                )

            agent = Agent(
                model=model,
                system_prompt=SYSTEM_PROMPT,
                tools=tools,
                hooks=[memory_hook],
            )

            response = await agent.invoke_async(user_input)

            for block in response.message.get("content", []):
                if "text" in block:
                    return block["text"]

            return "The agent completed without a text response."

    except Exception:
        logger.exception("Customer support invocation failed")
        return {
            "error": (
                "I could not complete your request. "
                "Please try again shortly."
            )
        }


# ── CLI entry point (do not modify) ──────────────────────────────────────────
def main():
    """Run one invocation from the command line for local testing."""
    parser = argparse.ArgumentParser()
    parser.add_argument("payload", type=str)
    args = parser.parse_args()
    response = asyncio.run(invoke(json.loads(args.payload)))
    print(response)


if __name__ == "__main__":
    app.run()
    # Uncomment the line below and comment app.run() for local CLI testing:
    # main()
