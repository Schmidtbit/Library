# library
Things I want to remember....


- [How to make function dependency packages for lambda functions](https://docs.aws.amazon.com/lambda/latest/dg/lambda-python-how-to-create-deployment-package.html#python-package-dependencies)

#### Upload a file to s3
```
import boto3
s3 = boto3.resource('s3')
s3.meta.client.upload_file('filename', 'mybucket', 'filename')
```
