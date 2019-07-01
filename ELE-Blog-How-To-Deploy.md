# How to Deploy a Machine Learning Risk Classification Model with Amazon AWS

In this article, I am going to walk through a high-level example of how to deploy a machine learning model using Amazon Web Services (AWS). The beauty of using AWS is that all of their services are pay-per-use with extremely competitive rates. For a data scientist just playing around and learning the AWS suite of services, there are many ways to implement model deployment architectures at minimal cost.

## Risk Modeling with Application Data 

The architecture proposed in this article is based upon a use case where a company (let’s call them Company X) collects a lot of user or application data coming from various sources. Below are all the ways in which Company X collects application data:

* __Webform__: Company X has a website where users log-in, create a profile and fill out a webform application. This is the most common method applicants use.
* __Mobile App__: Since there are more people in the world with a mobile phone than a computer, Company X decides to develop a mobile app service. This is a popular choice among a growing segment of applicants.
* __ChatBot__: Company X also has a chatbot service deployed on their website. Applicants submit their information to a bot agent via a chat dialog. The bot agent collects applicant information and provides an interface for answering commonly asked questions throughout the application process.
* __By Phone__: Applicants can phone in their application to a call center that utilizes both live agents and voice-bot agents. Voice-bot agents collect the bulk of the applicant data, but there are also live agents available to assist.
* __By Mail__: Applicants also have the option of printing out and submitting a paper application. The paper application is manually entered into the back-end system for processing.

At Company X, applicant data is ingested and temporarily stored in a DynamoDB NoSQL database. Applications are streamed and routed (via Amazon Lambda) into an Amazon S3 Bucket and an Amazon Redshift MySQL relational database. Amazon S3 Bucket storage is a great option for storing data that is frequently accessed and for performing fast ad hoc queries. Amazon Redshift is a relational MySQL database and is optimized for aggregate analytical queries and routine business analytics.

#### Example Architecture for Data Ingestion with Cloud Hosted Analytics on AWS

<img src="src/AnalyticsArchitecture.png" width="600"/>

## Machine Learning and the Risk Audititing Process 

Company X decides to develop a machine learning algorithm that scores applicants based on a three-tiered risk category (low, medium, and high risk). The categories are delineated based on the probability of an applicant achieving some desired (or undesired) state.

Risk modeling can be applied to many industries and in many different scenarios. Any time a company is interested in managing a particular outcome there is an opportunity to apply machine learning and advanced analytics to measure and predict the outcome. In the case of applicant data, you can achieve savings and efficiency by using machine learning algorithms to assist in making business decisions, such as whether or not an applicant should be accepted or rejected.

In this scenario, we want to train a model that will simulate the manual auditing process of accepting or rejecting an application. Applicant training data is tied to a label that marks the applicant status. The status can be a multi-tiered grade or a binary yes/no decision. The machine learning algorithm 'learns' over time how auditors make their decisions and attempt to mimic their decision-making process when given a new application.

## Step 1) Architect the Data Ingestion Pipeline using API Gateway

API Gateway is an easy and secure way to monitor and maintain your data ingestion process. It’s a pay-per-use service that keeps track of all your API communication and has a global reach. You can set up an API Gateway to receive data from all of your data ingestion touchpoints. Mobile and desktop clients, IoT devices, and bot services in voice, phone, and text are all sources of data that you can incorporate into your data ingestion architecture. 

In our example, we are interested in using Amazon API Gateway for its log data, which will provide additional business insights and additional metrics from which to build better and more robust models from.

## Step 2) Build a Model & Launch an Endpoint in SageMaker

Amazon SageMaker is a machine learning platform that is equipped with a ton of features designed to streamline the machine learning development and launch process. Machine learning is hard but creating the infrastructure to support the process doesn't have to be. Once you develop an algorithm in SageMaker, training and launching a model endpoint API in the cloud is as easy as writing a few lines of code.

SageMaker is a fully managed service that will handle all the messy business of provisioning containers and hosting your model endpoint for you. Amazon will take care of a number of tasks associated with provisioning a model into production and optimizing the required infrastructure according to your needs. For example:

 - [Inference Pipelines](https://docs.aws.amazon.com/sagemaker/latest/): Models are most often not standalone entities, but ensembles and stacks of data transformed multiple times over before the final result is infered. SageMaker  provides an easy way to orchistrate multiple inferences and transformations into a single pipeline.
 - [Auto Scaling](https://docs.aws.amazon.com/sagemaker/latest/dg/endpoint-auto-scaling.html): Set a CloudWatch alarm to trigger additional computing power by spinning up additional EC2 instances when you get a spike in model inference calls. This helps if you are interested in making real-time inferences and you have an unpredictible request load. 
 - [Elastic Inferece (EI)](https://docs.aws.amazon.com/sagemaker/latest/dg/ei.html): Do you have a computationally heavy model inference but don't like the high cost of hosting your endpoint on a GPU? Amazon EI is a service that will speed up the throughput of your deep learning models by provisioning an "accelerator" compute resource at the time the model is called. In many cases you can ditch the GPU.
 - [Neo](https://docs.aws.amazon.com/sagemaker/latest/dg/neo.html): Neo can optimize your model across TensorFlow, Apache MXNet, PyTorch, ONNX, and XGBoost for deployment on ARM, Intel, and Nvidia processors. Train your model once and use it anywhere. 
 
* [Inference Pipelines](https://docs.aws.amazon.com/sagemaker/latest/): Models are often not standalone entities, but ensembles and stacks of data transformed multiple times over before the final result is inferred. SageMaker provides an easy way to orchestrate multiple inferences and transformations into a single pipeline.
* [Auto Scaling](https://docs.aws.amazon.com/sagemaker/latest/dg/endpoint-auto-scaling.html): Set a CloudWatch alarm to trigger additional computing power by spinning up additional EC2 instances when you get a spike in model inference calls. This helps if you are interested in making real-time inferences and you have an unpredictable request load.
* [Elastic Inferece (EI)](https://docs.aws.amazon.com/sagemaker/latest/dg/ei.html): Do you have a computationally heavy model inference but don't like the high cost of hosting your endpoint on a GPU? Amazon EI is a service that will speed up the throughput of your deep learning models by provisioning an "accelerator" compute resource at the time the model is called. In many cases, you can ditch the GPU.
* [Neo](https://docs.aws.amazon.com/sagemaker/latest/dg/neo.html): Neo can optimize your model across TensorFlow, Apache MXNet, PyTorch, ONNX, and XGBoost for deployment on ARM, Intel, and Nvidia processors. Train your model once and use it anywhere.

Another benefit to using SageMaker is that it promotes transparency and collaboration by centralizing model development activities on a single platform. If you are a manager or owner and you are interested in monitoring your Data Science team's activities and utilization, then using SageMaker in conjunction with IAM, CloudWatch, and CloudTrail can give you insights into what data is being accessed, who (or what) is accessing that data, how it is being used, and how much are these activities costing.

If you are a data scientist, you will appreciate how easy it is to organize, annotate, and share your work with colleagues and managers by using the built-in jupyter notebooks. The environment you configure in the notebook stays in the cloud, making it easy to pick up where you left off or collaborate on a model without having to deal with environment configurations.

## Step 3) Create ETL Functions in Lambda

Amazon Lambda is a serverless compute option as part of Amazon's AWS compute suite of services. Think of them as the 'neurons' where the AWS solution architecture is this the 'brain'. Lambda functions don't require any provisioning of resources. They are on-the-fly pieces of code used for quick and simple function execution and routing of data.

In our example, we use Lambda to perform the ETL pre-processing steps required to make the data readable to the model endpoint. ETL stands for extract, transform, and load. We are extracting the data from the source (the API Gateway), transforming the data into a readable format for the model, and loading the data into the model endpoint with an API call.

## Step 4) Use DynamoDB as an Intermediate Database

Amazon's NoSQL database service, DynamoDB, is a great way to manage streaming and unpredictable data. All transactions are preserved because each call to Dynamo DB is a separate and recordable event. If you enable the 'streams' feature then you get the functionality of Amazon Kinesis Firehose, allowing you to handle spikes in traffic by turning your data streams into a parallel compute MapReduce job.

In our example, Company X writes data directly into a DynamoDB, which is a valid solution. However, if Company X wants to deploy a machine learning model inference into the application process, I would recommend doing so before the data is written to DynamoDB. Why? For two reasons: 1) because you can take advantage of real-time analytics on model inference calls, and 2) it gives you the option to provide immediate feedback to the user or client if you choose to. This may or may not be possible depending on the compute needs of the model inference, but it is preferable if the option is available to you.

#### Example Architecture for Real-Time Model Inferencing on AWS

<img src="src/Copy-multi-client-architecture2.png" width="600"/>

## Advantages & Benefits

By integrating a machine learning risk classification model at the front of the data ingestion pipeline, Company X is able to make rapid decisions, speed up their auditing process, and gain more granular insights into the application process.

## About ELE.ai
Interested in incorporating AI and machine learning into your business but don't know where to start or how to find a data scientist? Have a heavy data science workload and need additional resources to help bring your AI projects to furition? ELE.ai can help you start or scale your data science initiatives. Learn more at [ELE.ai](https://ele.ai/). 
