"""
Lambda API Handler (src/lambda_api.py)
---------------------------------------
Wraps the FastAPI app (app.py) for AWS Lambda + API Gateway
using Mangum adapter. Zero changes to app.py required.

Endpoints available via API Gateway:
    GET  /health
    GET  /appointments
    POST /appointments/book
    POST /process-s3-audio
    POST /process-s3-addendum
"""
from mangum import Mangum
from app import app

# Mangum wraps FastAPI → Lambda handler
handler = Mangum(app, lifespan="off")
