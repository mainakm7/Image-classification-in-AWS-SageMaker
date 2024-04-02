import boto3
import sagemaker
from sagemaker.session import Session
from sagemaker import get_execution_role
from sagemaker.debugger import Rule, rule_configs
from sagemaker.session import TrainingInput
from sagemaker.model import Model
from sagemaker.model_monitor import DataCaptureConfig
from sagemaker.serializers import IdentitySerializer
import base64
import json
import random


role = get_execution_role()
session = sagemaker.Session()
region = session.boto_region_name
bucket = session.default_bucket()


algo_image = sagemaker.image_uris.retrieve('image-classification',region,'latest')
s3_output_location = f"s3://{bucket}/l2project/models/image_model"


img_classifier_model=sagemaker.estimator.Estimator(
    image_uri=algo_image,
    sagemaker_session=session,
    role=role,
    instance_count=1,
    instance_type = "ml.p3.2xlarge",
    output_path=s3_output_location

)


img_classifier_model.set_hyperparameters(
    image_shape= "3,32,32", 
    num_classes= 2, 
    num_training_samples= len(df_train) 
)



model_inputs = {
        "train": sagemaker.inputs.TrainingInput(
            s3_data=f"s3://{bucket}/l2project/train/",
            content_type="application/x-image"
        ),
        "validation": sagemaker.inputs.TrainingInput(
            s3_data=f"s3://{bucket}/l2project/test/",
            content_type="application/x-image"
        ),
        "train_lst": sagemaker.inputs.TrainingInput(
            s3_data=f"s3://{bucket}/l2project/train.lst",
            content_type="application/x-image"
        ),
        "validation_lst": sagemaker.inputs.TrainingInput(
            s3_data=f"s3://{bucket}/l2project/test.lst",
            content_type="application/x-image"
        )
}

img_classifier_model.fit(model_inputs)

# image_uri = sagemaker.image_uris.retrieve(framework='image-classification',region=region)

# model_data = "s3://bucket/l2project/models/image_model/image-classification-2024-03-31-23-22-08-785/output/model.tar.gz"

# img_classifier_model = Model(image_uri=image_uri, model_data=model_data, role=role)


data_capture_config = DataCaptureConfig(enable_capture = True,
                                        sampling_percentage = 100,
                                        destination_s3_uri=f"s3://{bucket}/l2project/data_capture")

deployment = img_classifier_model.deploy(
    initial_instance_count=1,
    instance_type="ml.m5.xlarge",
    data_capture_config=data_capture_config
    )

endpoint = deployment.endpoint_name

predictor = sagemaker.predictor.Predictor(endpoint)

predictor.serializer = IdentitySerializer("image/png")
with open("./test/bicycle_s_001789.png", "rb") as f:
    payload = base64.b64encode(f.read())

    
inference = predictor.predict((base64.b64decode(payload)), initial_args={'ContentType': 'image/png'})



