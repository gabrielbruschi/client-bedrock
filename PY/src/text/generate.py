import boto3
from botocore.exceptions import ClientError
from config import BEDROCK_REGION_NAME, BEDROCK_MODEL_ID

# Create a Bedrock Runtime client in the AWS Region you want to use.
client = boto3.client("bedrock-runtime", region_name=BEDROCK_REGION_NAME)

# Start a conversation with the user message.
user_message = "Describe the purpose of a 'hello world' program in one line."
conversation = [
    {
        "role": "user",
        "content": [{"text": user_message}],
    }
]

try:
    # Send the message to the model, using a basic inference configuration.
    response = client.converse(
        modelId=BEDROCK_MODEL_ID,
        messages=conversation,
        inferenceConfig={"maxTokens": 1021, "temperature": 0.5, "topP": 0.9},
    )

    # Extract and print the response text.
    response_text = response["output"]["message"]["content"][0]["text"]
    print(response_text)

except (ClientError, Exception) as e:
    print(f"ERROR: Can't invoke '{BEDROCK_MODEL_ID}'. Reason: {e}")
    exit(1)


