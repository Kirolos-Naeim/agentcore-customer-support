import argparse
import ast
import json
from pathlib import Path

import boto3
import yaml

parser = argparse.ArgumentParser()
parser.add_argument("--agent", required=True)
args = parser.parse_args()

root = Path(__file__).resolve().parent

settings = {}
for node in ast.parse((root / "main.py").read_text()).body:
    if isinstance(node, ast.Assign):
        for target in node.targets:
            if isinstance(target, ast.Name):
                if target.id in {"KB_ID", "MEMORY_ID", "REGION"}:
                    settings[target.id] = ast.literal_eval(node.value)

region = settings["REGION"]
config = yaml.safe_load((root / ".bedrock_agentcore.yaml").read_text())
agent = config["agents"][args.agent]

role_arn = agent["aws"]["execution_role"]
role_name = role_arn.rsplit("/", 1)[-1]
account = role_arn.split(":")[4]
runtime_id = agent["bedrock_agentcore"]["agent_id"]

session = boto3.Session(region_name=region)

if session.client("sts").get_caller_identity()["Account"] != account:
    raise RuntimeError("AWS credentials and deployment accounts do not match.")

if agent["aws"]["region"] != region:
    raise RuntimeError("main.py and deployment regions do not match.")

control = session.client("bedrock-agentcore-control")
runtime = control.get_agent_runtime(agentRuntimeId=runtime_id)
if runtime["roleArn"] != role_arn:
    raise RuntimeError("Deployment configuration has the wrong execution role.")

session.client("bedrock-agent").get_knowledge_base(
    knowledgeBaseId=settings["KB_ID"]
)
control.get_memory(memoryId=settings["MEMORY_ID"])

memory_arn = (
    f"arn:aws:bedrock-agentcore:{region}:{account}:memory/"
    + settings["MEMORY_ID"]
)

policy = {
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": "bedrock:Retrieve",
            "Resource": (
                f"arn:aws:bedrock:{region}:{account}:knowledge-base/"
                + settings["KB_ID"]
            ),
        },
        {
            "Effect": "Allow",
            "Action": [
                "bedrock-agentcore:GetMemory",
                "bedrock-agentcore:RetrieveMemoryRecords",
                "bedrock-agentcore:CreateEvent",
            ],
            "Resource": [memory_arn, memory_arn + "/*"],
        },
        {
            "Effect": "Allow",
            "Action": [
                "bedrock-agentcore:StartBrowserSession",
                "bedrock-agentcore:StopBrowserSession",
                "bedrock-agentcore:GetBrowserSession",
                "bedrock-agentcore:UpdateBrowserStream",
                "bedrock-agentcore:ConnectBrowserAutomationStream",
            ],
            "Resource": (
                f"arn:aws:bedrock-agentcore:{region}:aws:"
                "browser/aws.browser.v1"
            ),
        },
        {
            "Effect": "Allow",
            "Action": [
                "bedrock-agentcore:StartCodeInterpreterSession",
                "bedrock-agentcore:InvokeCodeInterpreter",
                "bedrock-agentcore:StopCodeInterpreterSession",
            ],
            "Resource": (
                f"arn:aws:bedrock-agentcore:{region}:aws:"
                "code-interpreter/aws.codeinterpreter.v1"
            ),
        },
    ],
}

session.client("iam").put_role_policy(
    RoleName=role_name,
    PolicyName="CustomerSupportIntegrations",
    PolicyDocument=json.dumps(policy),
)

print("Permissions applied to:", role_name)
