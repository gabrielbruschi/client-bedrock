import boto3
import json
from config import BEDROCK_REGION_NAME, BEDROCK_MODEL_ID_EMBED

client = boto3.client(service_name="bedrock-runtime", region_name=BEDROCK_REGION_NAME)

fact = "The first moon landing was in 1969."
animal = "cat"

response = client.invoke_model(
    body=json.dumps(
        {
            "inputText": animal,
        }
    ),
    modelId=BEDROCK_MODEL_ID_EMBED,
    accept="application/json",
    contentType="application/json",
)

response_body = json.loads(response.get("body").read())
print(response_body.get("embedding"))