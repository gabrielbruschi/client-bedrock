import { BedrockClient, ListFoundationModelsCommand, GetFoundationModelCommand } from '@aws-sdk/client-bedrock'

const client = new BedrockClient({
    region: 'us-east-2'
});

async function listFoundationModels(){
    const response = await client.send(
        new ListFoundationModelsCommand({})
    )
    console.log(response)
}

async function getModelInfo(modelName: string){
    const response = await client.send(
        new GetFoundationModelCommand({
            modelIdentifier:modelName
        })
    )
    console.log(response)
}
//listFoundationModels();
getModelInfo('amazon.nova-micro-v1:0');