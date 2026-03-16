"""
WoundCare AI — CDK Stack (Lambda + Step Functions Architecture)
===============================================================
Provisions all AWS infrastructure for the WoundCare AI pipeline.

Architecture:
    S3 Upload → S3 Event → Step Functions →
        Lambda 1: Transcribe (Whisper)
        Lambda 2: Fetch Patient Info (Scriberyte)
        Lambda 3: LLM Parse (Gemini 3 Pro)
        Lambda 4: Generate HTML + Upload to S3

    API Gateway → Lambda 5: FastAPI (Mangum)

Cost: ~$8/month fixed (vs ~$76/month for ECS architecture)
"""
import aws_cdk as cdk
from aws_cdk import (
    Stack,
    Duration,
    RemovalPolicy,
    aws_s3 as s3,
    aws_s3_notifications as s3n,
    aws_lambda as lambda_,
    aws_apigateway as apigw,
    aws_stepfunctions as sfn,
    aws_stepfunctions_tasks as tasks,
    aws_iam as iam,
    aws_secretsmanager as secretsmanager,
    aws_logs as logs,
)
from constructs import Construct


class WoundCareStack(Stack):

    def __init__(self, scope: Construct, construct_id: str, **kwargs):
        super().__init__(scope, construct_id, **kwargs)

        # ─────────────────────────────────────────────
        # 0. RESOURCE TAGS — Cost Tracking
        # ─────────────────────────────────────────────
        cdk.Tags.of(self).add("project", "woundcare-ai")
        cdk.Tags.of(self).add("environment", "production")
        cdk.Tags.of(self).add("owner", "woundcare-team")
        cdk.Tags.of(self).add("cost-center", "woundcare-ai-pipeline")

        # ─────────────────────────────────────────────
        # 1. S3 BUCKET — Audio Inbox + Chart Archive
        # ─────────────────────────────────────────────
        # Switching to UAT bucket for 3-4 days testing
        bucket_name = "woundcare-bucket"
        s3_input_prefix = "woundcare"
        s3_output_prefix = "woundcare"

        # Use existing bucket
        bucket = s3.Bucket.from_bucket_name(self, "WoundCareBucket", bucket_name)

        # ─────────────────────────────────────────────
        # 2. SECRETS MANAGER — API Keys
        # ─────────────────────────────────────────────
        google_secret = secretsmanager.Secret(
            self, "GoogleApiKey",
            secret_name="woundcare/google-api-key",
            description="Google Gemini API key for LLM chart parsing",
        )
        openai_secret = secretsmanager.Secret(
            self, "OpenAiApiKey",
            secret_name="woundcare/openai-api-key",
            description="OpenAI API key for Whisper transcription",
        )
        scriberyte_secret = secretsmanager.Secret(
            self, "ScriberyteDbCredentials",
            secret_name="woundcare/scriberyte-db-credentials",
            description="Scriberyte MSSQL DB credentials for fetching patient info",
        )

        # ─────────────────────────────────────────────
        # 3. IAM ROLE — Lambda Execution Role
        #    Shared by all 5 Lambda functions
        # ─────────────────────────────────────────────
        lambda_role = iam.Role(
            self, "WoundCareLambdaRole",
            role_name="woundcare-lambda-role",
            assumed_by=iam.ServicePrincipal("lambda.amazonaws.com"),
            managed_policies=[
                iam.ManagedPolicy.from_aws_managed_policy_name(
                    "service-role/AWSLambdaBasicExecutionRole"
                ),
            ],
        )
        # S3 full access to woundcare bucket
        bucket.grant_read_write(lambda_role)

        # Secrets read access
        for secret in [google_secret, openai_secret, scriberyte_secret]:
            secret.grant_read(lambda_role)

        # Step Functions start execution (for S3 trigger Lambda)
        lambda_role.add_to_policy(
            iam.PolicyStatement(
                actions=["states:StartExecution"],
                resources=["*"],
            )
        )

        # ─────────────────────────────────────────────
        # 4. LAMBDA LAYER — Python Dependencies
        #    All shared deps bundled as a layer
        # ─────────────────────────────────────────────
        deps_layer = lambda_.LayerVersion(
            self, "WoundCareDepsLayer",
            layer_version_name="woundcare-deps",
            description="WoundCare AI Python dependencies",
            code=lambda_.Code.from_asset("../lambda_layer/"),
            compatible_runtimes=[lambda_.Runtime.PYTHON_3_11],
            removal_policy=RemovalPolicy.RETAIN,
        )

        # ─────────────────────────────────────────────
        # 5. SHARED LAMBDA CONFIG
        # ─────────────────────────────────────────────
        shared_env = {
            "S3_BUCKET_NAME": bucket.bucket_name,
            "S3_INPUT_PREFIX": s3_input_prefix,
            "S3_OUTPUT_PREFIX": s3_output_prefix,
            "AWS_ACCOUNT_ID": self.account,
        }

        shared_lambda_props = dict(
            runtime=lambda_.Runtime.PYTHON_3_11,
            code=lambda_.Code.from_asset("../"),          # project root
            role=lambda_role,
            layers=[deps_layer],
            environment=shared_env,
            log_retention=logs.RetentionDays.THREE_MONTHS,
            tracing=lambda_.Tracing.ACTIVE,             # X-Ray tracing
        )

        # ─────────────────────────────────────────────
        # 6. LAMBDA FUNCTIONS — Pipeline Steps
        # ─────────────────────────────────────────────

        # Step 1 — Transcribe audio via Whisper
        fn_transcribe = lambda_.Function(
            self, "FnTranscribe",
            function_name="woundcare-transcribe",
            handler="src.lambda_transcribe.handler",
            timeout=Duration.minutes(5),        # Whisper can take ~60s for long audio
            memory_size=1024,                   # Needs memory for audio processing
            description="Step 1: Transcribe audio via OpenAI Whisper",
            **shared_lambda_props,
        )

        # Step 2 — Fetch patient info from Scriberyte
        fn_patient_info = lambda_.Function(
            self, "FnPatientInfo",
            function_name="woundcare-patient-info",
            handler="src.lambda_patient_info.handler",
            timeout=Duration.seconds(30),
            memory_size=256,
            description="Step 2: Fetch patient info from Scriberyte API",
            **shared_lambda_props,
        )

        # Step 3 — LLM parse transcript via Gemini
        fn_parse = lambda_.Function(
            self, "FnParse",
            function_name="woundcare-parse",
            handler="src.lambda_parse.handler",
            timeout=Duration.minutes(3),        # Gemini can take ~30s
            memory_size=512,
            description="Step 3: Parse transcript via Gemini 3 Pro",
            **shared_lambda_props,
        )

        # Step 4 — Generate HTML chart and upload to S3
        fn_generate = lambda_.Function(
            self, "FnGenerate",
            function_name="woundcare-generate",
            handler="src.lambda_generate.handler",
            timeout=Duration.minutes(2),
            memory_size=512,
            description="Step 4: Generate HTML chart and upload to S3",
            **shared_lambda_props,
        )

        # Step 5 — API (FastAPI via Mangum)
        fn_api = lambda_.Function(
            self, "FnApi",
            function_name="woundcare-api",
            handler="src.lambda_api.handler",
            timeout=Duration.seconds(30),
            memory_size=512,
            description="FastAPI endpoints via Mangum adapter",
            **shared_lambda_props,
        )

        # ─────────────────────────────────────────────
        # 7. STEP FUNCTIONS — Pipeline Orchestration
        # ─────────────────────────────────────────────

        # CloudWatch log group for Step Functions
        sfn_log_group = logs.LogGroup(
            self, "SfnLogGroup",
            log_group_name="/woundcare/pipeline",
            retention=logs.RetentionDays.THREE_MONTHS,
            removal_policy=RemovalPolicy.DESTROY,
        )

        # Define pipeline steps
        step1 = tasks.LambdaInvoke(
            self, "Step1Transcribe",
            lambda_function=fn_transcribe,
            output_path="$.Payload",
            retry_on_service_exceptions=True,
        ).add_retry(
            max_attempts=3,
            interval=Duration.seconds(5),
            backoff_rate=2,
        )

        step2 = tasks.LambdaInvoke(
            self, "Step2PatientInfo",
            lambda_function=fn_patient_info,
            output_path="$.Payload",
        ).add_retry(
            max_attempts=3,
            interval=Duration.seconds(3),
            backoff_rate=2,
        )

        step3 = tasks.LambdaInvoke(
            self, "Step3Parse",
            lambda_function=fn_parse,
            output_path="$.Payload",
        ).add_retry(
            max_attempts=2,
            interval=Duration.seconds(10),
            backoff_rate=2,
        )

        step4 = tasks.LambdaInvoke(
            self, "Step4Generate",
            lambda_function=fn_generate,
            output_path="$.Payload",
        ).add_retry(
            max_attempts=3,
            interval=Duration.seconds(5),
            backoff_rate=2,
        )

        # Chain steps: 1 → 2 → 3 → 4
        pipeline_definition = step1.next(step2).next(step3).next(step4)

        state_machine = sfn.StateMachine(
            self, "WoundCarePipeline",
            state_machine_name="woundcare-pipeline",
            definition_body=sfn.DefinitionBody.from_chainable(pipeline_definition),
            timeout=Duration.minutes(15),
            tracing_enabled=True,
            logs=sfn.LogOptions(
                destination=sfn_log_group,
                level=sfn.LogLevel.ERROR,
            ),
        )

        # ─────────────────────────────────────────────
        # 8. S3 TRIGGER LAMBDA
        #    Fires when audio uploaded to woundcare/audio/
        #    Starts the Step Functions pipeline
        # ─────────────────────────────────────────────
        fn_trigger = lambda_.Function(
            self, "FnTrigger",
            function_name="woundcare-trigger",
            handler="src.lambda_trigger.handler",
            timeout=Duration.seconds(10),
            memory_size=128,
            description="S3 event trigger — starts Step Functions pipeline",
            environment={
                **shared_env,
                "STATE_MACHINE_ARN": state_machine.state_machine_arn,
            },
            **{k: v for k, v in shared_lambda_props.items()
               if k not in ["environment"]},
        )

        # Grant trigger Lambda permission to start Step Functions
        state_machine.grant_start_execution(fn_trigger)

        # S3 event notification → trigger Lambda on audio upload
        bucket.add_event_notification(
            s3.EventType.OBJECT_CREATED,
            s3n.LambdaDestination(fn_trigger),
            s3.NotificationKeyFilter(prefix=f"{s3_input_prefix}/"),
        )

        # ─────────────────────────────────────────────
        # 9. API GATEWAY — Public HTTP Endpoints
        # ─────────────────────────────────────────────
        api = apigw.LambdaRestApi(
            self, "WoundCareApi",
            rest_api_name="woundcare-api",
            handler=fn_api,
            description="WoundCare AI REST API",
            deploy_options=apigw.StageOptions(
                stage_name="prod",
                logging_level=apigw.MethodLoggingLevel.ERROR,
                data_trace_enabled=False,
                throttling_rate_limit=100,
                throttling_burst_limit=200,
            ),
            default_cors_preflight_options=apigw.CorsOptions(
                allow_origins=apigw.Cors.ALL_ORIGINS,
                allow_methods=apigw.Cors.ALL_METHODS,
            ),
        )

        # ─────────────────────────────────────────────
        # 10. SCRIBERYTE IAM USER
        #     Write-only access to woundcare/audio/ prefix
        # ─────────────────────────────────────────────
        scriberyte_user = iam.User(
            self, "ScriberyteS3User",
            user_name="scriberyte-audio-uploader",
        )
        scriberyte_user.add_to_policy(
            iam.PolicyStatement(
                sid="ScriberyteAudioUploadOnly",
                effect=iam.Effect.ALLOW,
                actions=["s3:PutObject", "s3:GetBucketLocation"],
                resources=[
                    f"{bucket.bucket_arn}/{s3_input_prefix}/*",
                    bucket.bucket_arn,
                ],
            )
        )
        scriberyte_access_key = iam.AccessKey(
            self, "ScriberyteAccessKey",
            user=scriberyte_user,
        )
        scriberyte_credentials_secret = secretsmanager.Secret(
            self, "ScriberyteS3Credentials",
            secret_name="woundcare/scriberyte-s3-credentials",
            description="AWS credentials for Scriberyte to upload audio to S3.",
            secret_object_value={
                "access_key_id": cdk.SecretValue.unsafe_plain_text(
                    scriberyte_access_key.access_key_id
                ),
                "secret_access_key": scriberyte_access_key.secret_access_key,
                "bucket_name": cdk.SecretValue.unsafe_plain_text(bucket.bucket_name),
                "region": cdk.SecretValue.unsafe_plain_text(self.region),
                "allowed_prefix": cdk.SecretValue.unsafe_plain_text(f"{s3_input_prefix}/"),
            },
        )

        # ─────────────────────────────────────────────
        # 11. CFN OUTPUTS — Shown after cdk deploy
        # ─────────────────────────────────────────────
        cdk.CfnOutput(self, "BucketName",
            value=bucket.bucket_name,
            description="S3 bucket for audio uploads and chart storage")

        cdk.CfnOutput(self, "ApiEndpoint",
            value=api.url,
            description="API Gateway URL — share with integrators")

        cdk.CfnOutput(self, "StateMachineArn",
            value=state_machine.state_machine_arn,
            description="Step Functions pipeline ARN")

        cdk.CfnOutput(self, "GoogleSecretArn",
            value=google_secret.secret_arn,
            description="Set Gemini API key here in Secrets Manager")

        cdk.CfnOutput(self, "OpenAiSecretArn",
            value=openai_secret.secret_arn,
            description="Set OpenAI API key here in Secrets Manager")

        cdk.CfnOutput(self, "ScriberyteApiKeyArn",
            value=scriberyte_secret.secret_arn,
            description="Set Scriberyte API key here in Secrets Manager")

        cdk.CfnOutput(self, "ScriberyteS3CredentialsArn",
            value=scriberyte_credentials_secret.secret_arn,
            description="Share these S3 credentials with Scriberyte team")

        cdk.CfnOutput(self, "ScriberyteAudioPrefix",
            value=f"s3://{bucket.bucket_name}/{s3_input_prefix}/",
            description="Scriberyte must upload audio files to this S3 path")
