"""
WoundCare AI - Infrastructure as Code Entry Point (infra/app.py)
-----------------------------------------------------------------
AWS CDK app that provisions all cloud infrastructure for the WoundCare AI pipeline.

Deploy with:
  cd infra
  pip install -r requirements.txt
  cdk bootstrap   # First time only
  cdk deploy
"""
import aws_cdk as cdk
from stacks.woundcare_stack import WoundCareStack

app = cdk.App()

WoundCareStack(app, "WoundCareStack",
    env=cdk.Environment(
        account=app.node.try_get_context("account"),
        region=app.node.try_get_context("region") or "us-east-1"
    ),
    description="WoundCare AI Pipeline — S3, ECS Fargate API + Watcher, ECR, IAM, Secrets"
)

app.synth()
