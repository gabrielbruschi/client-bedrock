import boto3
import pprint
from config import BEDROCK_REGION_NAME, BEDROCK_MODEL_ID

bedrock = boto3.client(
    service_name='bedrock',
    region_name=BEDROCK_REGION_NAME)

pp = pprint.PrettyPrinter(depth=4)

def list_foundation_models():
    models = bedrock.list_foundation_models()
    for model in models["modelSummaries"]:
        pp.pprint(model)
        pp.pprint("-------------------")

def get_foundation_model(modelIdentifier):
    model = bedrock.get_foundation_model(modelIdentifier=modelIdentifier)
    pp.pprint(model)

#list_foundation_models()
get_foundation_model(BEDROCK_MODEL_ID)
