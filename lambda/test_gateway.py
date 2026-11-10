import json
import uuid

import boto3
from mcp.client.streamable_http import streamable_http_client
from strands.tools.mcp.mcp_client import MCPClient

control = boto3.client("bedrock-agentcore-control", region_name="us-east-1")

gateways = []
request = {}
while True:
    page = control.list_gateways(**request)
    gateways.extend(page.get("items", []))
    token = page.get("nextToken")
    if not token:
        break
    request = {"nextToken": token}

matches = [
    gateway for gateway in gateways
    if gateway["name"] == "CustomerSupportGateway"
]
if len(matches) != 1:
    raise RuntimeError(
        "Expected one gateway named CustomerSupportGateway. "
        f"Found: {[g['name'] for g in gateways]}"
    )

gateway = control.get_gateway(
    gatewayIdentifier=matches[0]["gatewayId"]
)
url = gateway["gatewayUrl"]
print("Gateway URL:", url)

client = MCPClient(lambda: streamable_http_client(url=url))

with client:
    tools = client.list_tools_sync()
    names = [tool.tool_name for tool in tools]

    print("\nDiscovered tools:")
    for name in names:
        print("-", name)

    def call(tool_name, arguments):
        matches = [
            name for name in names
            if name.rsplit("___", 1)[-1] == tool_name
        ]
        if len(matches) != 1:
            raise RuntimeError(f"Cannot uniquely find tool: {tool_name}")

        print(f"\nCalling {matches[0]}:")
        result = client.call_tool_sync(
            tool_use_id=str(uuid.uuid4()),
            name=matches[0],
            arguments=arguments,
        )
        print(json.dumps(result, indent=2, default=str))

    call("get_order", {"order_id": "ORD-001"})

    call("initiate_refund", {
        "order_id": "ORD-002",
        "reason": "Course project test",
        "amount": 139.99,
    })
