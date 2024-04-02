#We make 3 separate Lambda functions to link together for StepFunction call


# Lambda1

import json
import boto3
import base64

s3 = boto3.client('s3')

def lambda_handler(event, context):
    """A function to serialize target data from S3"""
    
    
    key = event["s3_key"]         
    bucket = event["s3_bucket"]     
    
    
    s3.download_file(bucket,key,"/tmp/image.png")
    
    
    
    with open("/tmp/image.png", "rb") as f:
        image_data = base64.b64encode(f.read())

    
    print("Event:", event.keys())
    return {
        'statusCode': 200,
        'body': {
            "image_data": image_data,
            "s3_bucket": bucket,
            "s3_key": key,
            "inferences": []
        }
    }


#Lambda2

import json
import base64
import boto3



ENDPOINT = "image-classification-2024-04-01-18-38-28-650" 
runtime = boto3.client("runtime.sagemaker")


def lambda_handler(event, context):

    
    image = base64.b64decode(event["body"]["image_data"])

    
    
    inferences = runtime.invoke_endpoint(EndpointName=ENDPOINT,
                                         ContentType="image/png",
                                         Body=image)["Body"].read().decode("utf-8")
    
    
    event["inferences"] = json.loads(inferences)
    return {
        'statusCode': 200,
        'body': {
            "image_data": event["body"]["image_data"],
            "s3_bucket": event["body"]["s3_bucket"],
            "s3_key": event["body"]["s3_key"],
            "inferences": event["inferences"]
        }
    }


#Lambda3

import json


THRESHOLD = .8


def lambda_handler(event, context):
    
    
    inferences = event["body"]["inferences"]  
    
    
    meets_threshold = THRESHOLD  
    
    
    if meets_threshold:
        pass
    else:
        raise("THRESHOLD_CONFIDENCE_NOT_MET")

    return {
        'statusCode': 200,
        'body': {
            "image_data": event["body"]["image_data"],
            "s3_bucket": event["body"]["s3_bucket"],
            "s3_key": event["body"]["s3_key"],
            "inferences": event["body"]["inferences"]
        }
    }