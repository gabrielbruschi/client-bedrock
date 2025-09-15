import boto3
from botocore.exceptions import ClientError
from config import BEDROCK_REGION_NAME, BEDROCK_MODEL_ID

client = boto3.client(service_name='bedrock-runtime', region_name=BEDROCK_REGION_NAME)

user_message = """
**Instrução:**

Você é um assistente especialista em extração de dados de documentos financeiros, focado em precisão absoluta. Sua tarefa é analisar o texto de um recibo de comissões e preencher o esqueleto JSON fornecido.

**REGRAS OBRIGATÓRIAS:**
1.  **EXTRAÇÃO ESTRITA:** Extraia os valores *exatamente* como aparecem no texto. Não calcule, infira, deduza ou invente nenhuma informação.
2.  **DADOS AUSENTES:** Se uma informação ou valor não for encontrado explicitamente no texto, o valor correspondente no JSON DEVE ser `null`. Não preencha com "0" ou com valores do esqueleto.
3.  **CONTEXTO DE PAGAMENTO:** Para valores de tributos (IRRF, ISSQN), utilize sempre os dados da coluna ou seção "No Pagamento", e não da coluna "Acumulado".
4.  **FORMATAÇÃO:** Converta valores monetários para o tipo numérico (float), removendo "R$" ou "BRL". Datas devem seguir o formato "DD/MM/AAAA".
5.  **SAÍDA LIMPA:** Responda APENAS com o objeto JSON final, sem nenhum texto, explicação ou ```json ``` no início ou fim.

**Guia de Extração por Campo (Dicas Semânticas):**
* `numero_extrato`: Pode aparecer como "Número da Fatura" ou "Número do Extrato".
* `data_extrato`: É a data de referência do extrato, geralmente aparece no topo como "Extrato de Pagamento de...".
* `nome_corretor`: Geralmente identificado pelo rótulo "PARCEIRO" ou "Corretor".
* `total_comissao_bruta`: Procure por "Valor Bruto", "Créditos" ou um total similar antes das deduções de impostos.
* `total_comissao_liquida`: Procure por "Valor Líquido" ou "Valor de Crédito*".

**Esqueleto JSON:**
{
    "numero_extrato": "string",
    "data_extrato": "string",
    "nome_corretor": "string",
    "susep_corretor": "string",
    "cpfcnpj_corretor": "string",
    "cod_corretor": "string",
    "total_comissao_bruta": "float",
    "aliquota_iss": "float",
    "aliquota_inss": "float",
    "aliquota_irrf": "float",
    "valor_iss": "float",
    "valor_inss": "float",
    "valor_irrf": "float",
    "total_comissao": "float",
    "total_comissao_liquida": "float"
}

**Texto a ser processado:**
[AQUI VOCÊ INSERE O TEXTO DO RECIBO]
"""

conversation = [
    {
        "role": "user",
        "content": [{"text": user_message}],
    }
]

try:
    response = client.converse(
        modelId=BEDROCK_MODEL_ID,
        messages=conversation,
        inferenceConfig={"maxTokens": 1024, 
                         "temperature": 0.0, 
                         "topP": 1.0},
    )

    response_text = response["output"]["message"]["content"][0]["text"]
    print(response_text)

except (ClientError, Exception) as e:
    print(f"ERROR: Can't invoke '{BEDROCK_MODEL_ID}'. Reason: {e}")
    exit(1)