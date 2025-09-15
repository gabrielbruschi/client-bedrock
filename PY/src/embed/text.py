import boto3
import json
from config import BEDROCK_REGION_NAME, BEDROCK_MODEL_ID_EMBED
from src.embed.similarity import cosineSimilarity

client = boto3.client(service_name='bedrock-runtime', region_name=BEDROCK_REGION_NAME)

facts = [
    'The first computer was invented in the 1940s.',
    'John F. Kennedy was the 35th President of the United States.',
    'The first moon landing was in 1969.',
    'The capital of France is Paris.',
    'Earth is the third planet from the sun.',
]

question = 'Who is the president of USA?'

def getEmbedding(input: str):
    response = client.invoke_model(
        body=json.dumps({
            "inputText": input,
        }), 
        modelId=BEDROCK_MODEL_ID_EMBED, 
        accept='application/json', 
        contentType='application/json')

    response_body = json.loads(response.get('body').read())
    return response_body.get('embedding')

factsWithEmbeddings = []

for fact in facts:
    factsWithEmbeddings.append({
        'text': fact,
        'embedding': getEmbedding(fact)
    })

newFactEmbedding = getEmbedding(question)

similarities = []

for fact in factsWithEmbeddings:
    similarities.append({
        'text': fact['text'],
        'similarity': cosineSimilarity(fact['embedding'], newFactEmbedding)
    })

print(f"Similarities for fact: '{question}' with:")
similarities.sort(key=lambda x: x['similarity'], reverse=True)
for similarity in similarities:
    print(f"  '{similarity['text']}': {similarity['similarity']:.2f}")
