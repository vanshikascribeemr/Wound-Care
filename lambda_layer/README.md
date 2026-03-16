# Lambda Layer Dependencies
This folder is used to bundle the Python dependencies for the WoundCare AI Lambda functions.

### Instructions to Build:
1. Ensure you have Docker or a Linux environment (for cross-compilation).
2. Run the following command from the project root:
   ```bash
   pip install -r requirements.txt -t lambda_layer/python/
   ```
3. Alternatively, use the `build_layer.sh` script if provided.
4. The CDK stack expects this folder to exist during `cdk deploy`.
