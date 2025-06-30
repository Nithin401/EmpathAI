from moto import mock_aws
import boto3
import json

@mock_aws
def check_s3():
    s3_client = boto3.client('s3')
    response = s3_client.list_objects_v2(Bucket='empathai-chat-sessions', Prefix="chat_sessions/")
    if 'Contents' in response:
        for obj in response['Contents']:
            obj_data = s3_client.get_object(Bucket='empathai-chat-sessions', Key=obj['Key'])
            print(f"Found: {obj['Key']}, Data: {obj_data['Body'].read().decode('utf-8')}")
    else:
        print("No chat sessions found.")

check_s3()