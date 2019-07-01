# How to Write an AWS Lambda Handler Script

```
import requests
import boto3

s3 = boto3.resource('s3')

def lambda_handler(event, context):
    print("downloading Travis County Deliquent Tax Data to sagerealestate")
   
    url = 'downloadable-url-link'
    r = requests.get(url, allow_redirects=True)
    file_key = 'directory/filename'
    s3.Bucket('bucket_name').put_object(Key=file_key, Body=r.content)
    print('done')
    
```
# How To Create Lambda Function Dependency Packages
- [Link to AWS Docs.](https://docs.aws.amazon.com/lambda/latest/dg/lambda-python-how-to-create-deployment-package.html#python-package-dependencies)

 
