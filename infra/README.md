# WoundCare AI — Infrastructure (CDK)

> Architecture: **Lambda + Step Functions**
> Cost: **~$8/month fixed**

## Structure

```
infra/
├── app.py                  # CDK app entry point
├── cdk.json                # CDK config
├── requirements.txt        # CDK Python dependencies
└── stacks/
    └── woundcare_stack.py  # All AWS resources defined here
```

## Resources Defined

- **S3 Bucket** — audio inbox + chart storage
- **6 Lambda Functions** — trigger, transcribe, patient_info, parse, generate, api
- **Step Functions** — pipeline orchestration with retries
- **API Gateway** — public REST API
- **Lambda Layer** — shared Python dependencies
- **Secrets Manager** — API keys (Google, OpenAI, Scriberyte)
- **IAM** — Lambda role + Scriberyte write-only user
- **CloudWatch** — log groups + X-Ray tracing

## Deploy

```bash
# 1. Build Lambda layer first (from project root)
mkdir -p lambda_layer/python
pip install -r requirements.txt --target lambda_layer/python \
  --platform manylinux2014_x86_64 --implementation cp \
  --python-version 3.11 --only-binary=:all:

# 2. Install CDK deps
cd infra && pip install -r requirements.txt

# 3. Bootstrap (one-time)
cdk bootstrap aws://ACCOUNT_ID/REGION

# 4. Deploy
cdk deploy
```

See `CLOUD_ENGINEERING.md` in the project root for the full deployment guide.
