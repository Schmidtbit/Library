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
 
