import json
import boto3


sagemaker_runtime = boto3.client('sagemaker-runtime')

def lambda_handler(event, context):

    body = json.loads(event['body'])
    prompt = body['prompt']

    endpoint_name = "<REPLACE WITH YOUR ENDPOINT NAME>"

    #prompt = "what is deep learning"

    payload = {
    "inputs":prompt,
    "do_sample":True,
    "top_p":0.1,
    "temperature": 0.1,
    "top_k":30,
    "max_new_tokens":2048,
    "repetition_penalty":1.03,
    "return_full_text":False,
    "stop":["</s>"]
    }

    try:
        response = sagemaker_runtime.invoke_endpoint(
            EndpointName = endpoint_name,
            ContentType = "application/json",
            Body = json.dumps(payload)
        )
     

        response_body = json.loads(response['Body'].read().decode())

        generated_text = response_body[0]['generated_text']


        return {
            'statusCode':200,
            'body': json.dumps({'generated_text':generated_text})
        }
    except Exception as e:
        print(f"error happend {e}")

        return {
            'statusCode':500,
            'body':json.dumps({'error':e})
        }


