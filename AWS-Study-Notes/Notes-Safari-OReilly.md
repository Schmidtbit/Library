# Data Engineering

## Data Ingestion

### Data Lakes (S3 + AWS Glue)
- Handle structured and unstructured data
- Handle Analytics & ML
- Work on data without data movement
- A low cost storage option

___AWS Lake Formation___
1. ingest and register data
2. design access controls
3. collaborate and use data without having to worry about security and permissions 

### Kinesis Streaming
Good for real-time data workflows like Time Series Analytics and Real-time dashboards & metrics. "A series of pipes"

1. Select your input data stream (Kinesis Firehose)
2. write SQL queries to process data (Kinesis Analytics)
3. Configure output stream and destination 

__Example 1: Real-time Log Analytics (with clickstream data)__
1. Clickstream data is collected from websites
2. Kinesis Firehose ingests the logs
3. Kinesis Analytics queries the data
4. Outputs are sent to multiple destinations:
	- Redshift: for data warehousing
	- S3: for datalake
	- Elastic Search: to enable search capabilities

__Example 2: Ad Tech Pipeline__
1. data originates from an Ad server
2. Kinesis Streams ingests the data 
3. Kinesis Analytics is used for SQL analytics
4. Lambda is triggered (a single event function)
5. Real-time Ad placement System


__Example 2: Kinesis & IoT__
1. Data is created from an IoT devise
2. Kinesis Streams ingests the data
3. Kinesis Analytics queries the data
4. DynamoDB stores the data
5. Customers get dashboards and alerts in real time


___Kinesis Video Streams___:

 Securely stream video from connected devices to AWS for analytics, machine learning (ML), playback, and other processing.
- Smart home video streams
- Smart City (Amber Alert System)
- Industrial Automation (thermal imaging)

___Kinesis Data Streams___: 

KDS can continuously capture gigabytes of data per second from hundreds of thousands of sources such as website clickstreams, database event streams, financial transactions, social media feeds, IT logs, and location-tracking events. (Real Time)
- Log & Event Data Collection
- real-time Analytics
- mobile data capture
- Gaming Data Feed

___Kinesis Firehose___:

The easiest way to reliably load streaming data into data lakes, data stores and analytics tools. It can capture, transform, and load streaming data into Amazon S3, Amazon Redshift, Amazon Elasticsearch Service, and Splunk, enabling near real-time analytics with existing business intelligence tools and dashboards you’re already using today. You can also configure your delivery stream to automatically convert the incoming data to columnar formats like Apache Parquet and Apache ORC, before the data is delivered to Amazon S3, for cost-effective storage and analytics. (Near Real Time)
- IoT Analytics
- Clickstream Analytics
- Log Analytics
- Security Monitoring

___Kinesis Data Analytics___:

SQL users can easily query streaming data or build entire streaming applications using templates and an interactive SQL editor. Java developers can quickly build sophisticated streaming applications using open source Java libraries and AWS integrations to transform and analyze data in real-time. (Real Time)
- Streaming ETL for IoT with Java Apps
- Real-time log analytics with SQL
- Ad Tech & Digital Marketing with SQL


### AWS Batch
Designed to process data once or multiple times a day. 

__Example 1: Financial Trade Analytics__
1. Data is ingested into S3 
2. AWS Batch configures resources and schedules a batch analytic job
3. Results are sent back to S3

__Example 2: Film Studio Blueprints__
1. Artists create graphic blueprints throughout the day (blueprints pile up) and upload to S3
2. A render job is scheduled in the pipeline manager and AWS Batch fetches all the blueprints from S3
4. The blueprints are then rendered into hi-res files via a distributed batch job
5. Rendered files are sent back to S3


### Lambda (Events) 
Serverless Functions

__Services that use Lambda__:
- DeepLens
- SageMaker
- S3

___Step Functions in Lambda___
Control functions that have multiple lambdas. Has the ability to communicate with many services in the AWS ecosystem.

## Data Cleaning and Preparation
High Quality Data 
- validity
- accuracy
- completeness
- uniformity

___Amazon Services for Data Prep___
* AWS Glue
* Amazon Machine Learning (the service)
* Kinesis
* Lambda
* SageMaker

## Data Storage
* __Relational__: (Amazon Aurora, Amazon RDS) Referential integrity with strong consistency, transactions, and hardened scale
* __Key-Value__: (DynamoDB) Low latency, key based queries with high throughput and fast ingestion of data
* __Document__: (DynamoDB) Indexing and storing documents with support for queries on any property
* __Graph__: (Neptune) Creating and navigating relations between data easily and quickly
* __In-memory__: (Amazon ElastiCache for Redis & Memcashed) Microsecond latency, key-based queries, specialized data structures (good for ML recommendation algorithm)
* __Search__: (Amazon ES) Indexing and searching semi-structured logs and data

### Amazon S3
- connect to on-premise with __AWS Gateway__
- use __AWS Data Sync__ to automate and speed up transfers
- use __AWS Transfer for SFTP__ to enable secure file exchanges with 3rd parties
- What are the S3 Storage Types?
	- __S3 Standard__: Frequently accessed data
	- __S3 Intelligent Tiering__: For data with changing or unknown access patterns)
	- __S3 Standard-Infrequent Access (Standard - IA)__ and __One Zone-Infrequent Access (One Zone - IA)__: For data that is long lived but less frequently accessed
	- __S3 Glacier__ and __Glacier Deep Archive__: For long term archiving and digital preservation)

### Amazon DynamoDB
- Fully managed, multiregion, multimaster key-value and document database 
- built-in security, backup and restore
- in-memory caching for internet-scale applications
- Can handle more than 10 trillion requests per day 
- Can support peaks of more than 20 million requests per second

How to use & set-up defaults:
- set up a primary key structure
- default is set to have no secondary keys
- default is set to 5 reads/write at a time
- dafault is set to 80% upper threshold alarms using SNS 'dynamodb'
- Default Encryption at Rest with DEFAULT encryption type

___Applications___:
- Serverless Web Apps
- Mobile Backends
- Microservices


### Amazon Redshift 
- Data warehouse solution for AWS 
- Column data store (great at counting large data)
- compute optmized with machine learning
- concurrency scaling
- $0.25/hour and $250/TB/year
- Analyze S3 Data Lakes without ETL (Redshift Spectrum)
- Encryption using AWS-KMS or HSM
- Isolate clusters in VPCs
- Compliant with: SOC1, SOC2, SOC3, PCI DSS Level 1, FedRAMP, HIPPA (with BAA)

### Amazon RDS
* Available with:
	- Amazon Aurora
	- PostgreSQL
	- MySQL
	- MariaDB
	- OracleDB
	- SQL Server
* Read-Replicas provide capacity for read-heavy database workloads
* Multi-AZ deployment enhance availability and reliability
* Automated backups
* Database snapshots
* automatic host replacement

### Amazon ElastiCashe

Fully managed Redis and Memchashed (in-memory data store). Amazon ElastiCache is a popular choice for Gaming, Ad-Tech, Financial Services, Healthcare, and IoT apps.

* For low latency and high throughput
* Highly availabile & reliable
* Amazon VPC, HIPPA eligible, Encryption in transit/rest, redis Auth
* Auto-scales

### Amazon ES

Fully managed service that makes it easy for you to deploy, secure, and operate Elasticsearch at scale with zero down time. The service offers open-source Elasticsearch APIs, managed Kibana, and integrations with Logstash and other AWS Services, enabling you to securely ingest data from any source and search, analyze, and visualize it in real time. 

* good for machine generated data (pedabytes of data)
* auto-scales
* high availability when deployd across multi AZs
* daily backups
* built in encryption
* VPC and Cognito, and IAM security capabilities
* real-time analytic capabilities

Use Cases:
- Log Analytics
- Real-time Application Monitoring
- Security Analytics
- Full text search
- clickstream analytics

Integrates with:
- Elasticsearch APIs
- Kibana dashboards
- Logstash
- Kinesis Firehose
- AWS IoT
- CloudWatch Logs
- CloudTrail
- VPCs
- AWS KMS
- Amazon Cognito
- AWS IAM


## ETL (Extract-Transform-Load)

### AWS Glue

Fully Managed ETL service that detects schemas from relational DBs or S3, can transform that data and send to somewhere else. 
1. builda a data catalog
2. generate & edit transformations
3. schedule & run job

Integrates with:
* Athena (query)
* EMR 
* Redshift Spectrum

### Amazon EMR

For large-scale distributed data jobs. (i.e. Spark jobs)
- Run and scale Hadoop clusters (HDFS) of EC2 instances
	- Run applications based on: MapReduce, Spark, Presto, Pig, Hive, Impala, Hbase, Flink
- Can process structured, unstructured, and semi-structured data
- Integrates with AWS services
 	- Kinesis
	- DynamoDB
	- Redshift
	- Data Pipeline
	- S3
- Enables big data use cases

### Amazon Athena

Interactive query service that makes it easy to analyze data in Amazon S3 using standard SQL
- can replace many ETL (can query S3 directly)
- serverless (competitor to EMR)
- built on Presto w/ SQL Support
- meant to query a Data Lake

_** Works with AWS Glue_


### Data Pipeline

Useful for a batch-based ETL pipeline

__Capabilities__:
- Run AWS CLI comand or ShellCommand
- Import/Export DynamoDB table to/from S3
- Run johb on EMR cluster
- Full/Incremental copy of RDS MySQL table to S3 or Redshift
- load S3 data into RDS MySQL table or Redshift

### Streaming vs. Batch
- In Batch, more control of model training (can decide when to re-train)
- In Real-time (online), model could provide better prediction results or worse results:
	- did input stream suddenly get more users or less users?
	- Is there an A/B testing scenario?

__Batch__
- Data is batched at intervals
- Simplest approach to create predictions
- Serices on AWS that support batch processing:
	- AWS GLue
	- AWS Data Pipeline
	- AWS Batch
	- EMR

__Streaming__
- Continuously pulled/pushed
- more complex method of prediction
- Services on AWS that support streaming processing:
	- Kinesis
	- IoT

### Data Security, Backup, & Recovery

__KMS__ (Key Managemnet Service)
- Integrated with AWS Encryption SDK
- Integrated with Cloud Trail tells you who has access to encrypted data and their activity

 __CloudTrail__ 
- Enables governance, compliance, and operational auditing
- visibility to user access and activity
- security analysis and troubleshooting 
- can SQL query in Athena


__Data Backup & Recovery__

How do each of the storage services backup and snapshop?
- RDS
	- automatic backup and snapshotting
	- encryption at rest and in transit (with Transparent Data Encryption in SQL Server and Oracle)
	- integrates with AWS CloudHSM (Hardware Security Module)
	- Supports SSL to secure data in transit
	- May be run in a VPC network
	- Control user access with IAM permissions
- S3
	- Snapshot (versioning), Backup, & restore capabilities
	- ability to keep all versions of an object in the same bucket
	- tags track projects
	- object level logging
	- default encryption: AES256 or AWS-KMS
- DynamoDB 
	- Fully managed, multiregion, multimaster key-value and document database 
	- built-in security, backup and restore
	- in-memory caching for internet-scale applications
	- Can handle more than 10 trillion requests per day 
	- Can support peaks of more than 20 million requests per second


---
# EDA - Data Analysis on AWS

### Data Visualization
- QuickSight
- Plotly
- Clustering (used as a data viz tool as opposed to a unsupervized ML model in intself)
	- important to remember to normalize the data (preprocessing) to eliminate distortions
	- can be used to "discover" the labels
- Elbow Plot = diagnostic tool for clustering used to identify ideal number of clusters
	

The EDA Cycle:
- used to detect outliers
- see the data distribution

### Summary Stastistics (part of EDA Process)
- median
- max
- percentiles (25%, 50%, 75%, 10%)

### Heatmaps
- yellowbrick
- used to do feature ranking
- used to identify correlations

### PCA (Principal Component Analysis)
- reduces the dimentions in data without losing information
- available in SageMaker & SkLearn
- can reduce data down to 2-3 dimentions in order to plot data and visualize any outliers

### Data Distributions
- boxplots
- histograms


### Data Normalization
- Allows a non-distorted aspect of the data to emerge
- normalizes the magnitude of the data
- used as a preprocessing step

---

# ML Modelling in AWS

### What is ML?
- A subset of AI (and deep learning is a subset of ML)
- Types of ML: 
	- Supervized (has a label and must be retrained)
	- unsupervized (no label, instead it makes inferences about the data (i.e. clustering))
	- reinforcement (continuous self-taught learning based on agent-state relationship)
- Categories of ML: 
	- Regression (numbers)
 	- Classification (categorical)

### What can ML do?
- Make predictions
- Optimize utility functions
- Extract hidden data structures
- Classify data

### ML Use Cases

___Fraud Detection:___
	- mine data
	- identify hidden patterns and create labels (unsupervized learning)
	- train model
	- flag transactions as fraudulent

___Content Personalization:___ Use predictive analytics to recommend items (recommendation system)

___Targeted Marketing:___
- Use customer activity to choose revelant email campaigns
- cross-selling
- upselling

___Categorization:___
- Match hiring managers and resumes
- Categorized unstructured conetent/documents 

___Customer Service:___
- Analyze social media traffic to route customers to customer support
	- Predictive routing of customer emails

### ML Concepts 

___The ML Workflow:___
	- Clean Data (handle missing data, incorrect spelling, bad data, normalization, and making sure data is representative)
	- Create/train ML Model
	- Evaluate ML Model (using a holdout set)
	- Make predictions/inferences

___Predict:___ Predict new data based on observed data

___Extract:___ Extract hidden structure from data
	
___Summarize:___ Summarize data into an executive brief

___Adapt:___ Adapt based on observed data. 

___ML Types:___
- Supervized (regression/classification)
	- labels are known (for training)
	- Model "learns" to generalize from data (records)
	- Human needs to tell model which outputs are correct
	- Input --> Model --> Output/Prediction
	- Classification (a category like "red" or "blue")
	- Regression (numerical like 105.334)
- Unsupervized (clustering)
	- Labels are unknown
	- The labels are the "hidden structures" to be discovered
	- Self-organizing
	- Clustering
- Reinforcement (agent-state learning)
	- Model learns by interacting with environment
	- Learns to take action to maximize total reward
	- Inspired by behaviorial biology
	
### Frameworks & Infrastructures
- Apache MXNet
- Caffe and Caffe2
- TensorFlow

### ML Platforms
Fully managed platform for building models with own data:
- Apache Spark on EMR
- SparkML (25 petaNyte models, Zillow and Netflix)
- SageMaker
- Amazon Machine Learning

### API-driven Services
- Comprehend (NLP)
- Rekognition (Computer Vision)

### Models
- Can be stored in S3
- Can be A/B tested


### Optimized Instances & ML Images
- Deep Learning AMIs
- Containers

## Feature Engineering 
What are the attributes that are going to be used in an ML model?

A feature is an attribute used in a prediction model. ___Feature Engineering___ is the creation and curation of these attributes. 

__Example:__ Credit Applications

1. Pass in the dataFrame correlations to a heatmap and identify attributes that are correlated to one another. 
`sns.heatmap(df.corr())`

	You can also use Yellow Brick Road:
	
	```python
	from yellowbrick.features import Rank2D
	viz = Rank2D(algorithm='pearson', size=(1080,1080))
	viz.fit_transform(pd.get_dummies(df))
	viz.poof()
	```
2. Use information from the correlation heatmap plot and start training and testing out which features give better predicitons. 

	Or, use Amazons Machine Learning Service (AutoML):
	- Create Data Training Source
		- Prepare data 
		- create S3 bucket & folder
		- upload data to folder
		- give ML Service access to bucket
	- Schema is infered
	- Select Target
	- Review settings
	
## Model Training

1. Create 70/30 Split with sklearn 
2. One Hot Encode categorical variables
	
	```python
	from sklearn.model_selection import train_test_split
	from sklearn.feature_extraction import DictVectorizer
	
	dvec = DictVectorizer(sparse=False)
	vectorized_df = dvec.fit_transform(df.transpose().to_dict().values())
	train, test = train_test_split(vectorized_df, test_size=0.3)
	
	```
	
3. Use Logistic Regression

	```python
	from sklearn.linear_model import LogisticRegression
	
	X = train[:,:-1] # drop the label column
	Y = train[:,-1] # select only the label column
	model = LogisticRegression() # initialize model
	model.fit(X,Y) # fit the model
	```

## Model Evaluation

#### Classification Models
- Precision: (TP / TP + FP) - all true predictions that are actually true divided by the set of all predicted true 
- Recall: (TP / TP + FN) - all true predictions that are actually true, divided by the set of all true both predicted and actual 
- F1 (harmonic mean of precision and recall)
- AUC

Using Yellow Brick Road:
```python
from yellowbrick.classifier import ClassificationReport

viz = ClassificationReport(model)
viz.fit(X,Y)
viz.score(X,Y)
viz.poof()

```

Using SKLearn:
```python
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import precision_score, recall_score, roc_auc_score


y_train_pred = cross_val_predict(model, X, Y, cv=3)

# precision
precision_score(Y, y_train_pred)
# recall
recall_score(Y, y_train_pred)
# F1
f1_score(Y, y_train_pred)
# AUC
roc_auc_score(Y, ,_train_pred)
```

## Understanding ML Inference

### AWS ML Service
- inference referrs to prediction
- SageMaker can auto-scale inference endpoints
- ___Amazon Elastic Inference Service___
	- Can save up to 75% of cost of inference (prediction)
	- Attaches to SageMaker and EC2


## Understanding Deep Learning

### History of AI
- 1950-1998 Academic Activity
- 1998-2007 "AI Winter"
- 2007-Today "The GPU Era"
- Deep Learning is a "Multi-layered" feed-forward neural network

### Prefect Storm for Deep Learning
- cloud computing
- End of Moore's Law (2005)
- Elastic Storage & Compute
- Proliferation of Data

### Deep Neural Network (DNN)

__Layers__
Layers are simple functions than can be collectively trained (in layers) to learn complex functions
- input layer
- hidden layers ("Deep" Neural Nets)
- output layer

### Recurrent Neural Networks (RNNs)
Feedback loops at each "neuron" or node within the Neural Net
- good for:
	- text, speech, translation
	- self driving cars
	- time-series

### Deep Learning Use Cases

___Text Analysis using NLP:___
- insider trading
- sentiment analysis
- regulatory compliance
- brand affinity
- intent analysis

___Time Series:___
- financial 
- log analysis
- supply chain
- IoT

___Recommendation Engines:___
- Social Media

### Deep Learning AMIs
- Three Types:
	- Conda-based AMI
	- Base AMI
	- AMIs with source code
- Pay-as-you-go pricing
- Spot instance available
- Perform multi-GPU training 
- Pre-loaded with frameworks


---
# Operationalize ML on AWS

## Understand ML Operations
- Monitoring
	- Do you have enough resources to train?
	- Are you keeping an eye on CPU utilization? 
- Security
	- Does your team have appropriate access?
	- Is data protected or is there a security vulnerability?
- Retraining models
	- Is the model "online" real-time streaming predictions?
	- Is it a batch run model?
- A/B testing
- TCO
	- Able to scale up/down?
	- Picking right CPU/GPU configuration
	

### ML Ops
- Are you using a simple enough model?
- Are you using the Data Lake or wired directly into production SQL DB?
- Do you have alerts setup for prediction threshold failures?
- What are the environments for Dev, Stage, and Production?

### Containerization
Containers allow you to package your code and control the deployment environment

Containers can be deployed on:
- Elastic Container Service (based on docker)
- Elastic Kuberneties Service
- EC2 & AWS Cloud
- On-premise

##### Create a Docker Container with ECS:
1. Authrnticate your docker client to your regritry using AWS CLI
2. Build your Docker image 
3. After the build completes, tag your image so you can push to the repository
4. Push the image to your AWS repository

	```bash
	aws ecr get-login --no-include-email --region us-east-1

	docker build -t image

	docker tag mlimage:latest XXXXXXXXXXXX.dkr.ecr.us-east-1.amazonaws.com/mlimage:latest

	docker push XXXXXXXXXX.dkr.ecr.us-east-1.amazonaws.com/mlimage:latest
	```

## Implementing Continuous Deployment and Delivery for ML on AWS

__Code Pipeline:__ 

(Integrates with : ECS (Docker), GitHub, S3, and more...)
Code changes are automatically built, tested, and prepared for release to production. With AWS CloudFormation and CodePipeline, you can use continuous delivery to automatically build and test changes to your AWS CloudFormation templates before promoting them to production stacks. This process lets you rapidly and reliably make changes to your AWS infrastructure.

__Continuous delivery__ 

Automates the entire software release process. Every revision that is committed triggers an automated flow that builds, tests, and then stages the update. The final decision to deploy to a live production environment is triggered by the developer.

## Understanding A/B Testing Production Deployment
- SageMaker A/B testing Capabilities
- Deciding on ratio of delivery to ML Model

1. Go to Endpoint Configuration  
2. Create a new configuration
3. Select 2 models
4. Edit Production Variant Weight

>If you are hosting multiple models, you also assign a VariantWeight to specify how much traffic you want to allocate to each model. For example, suppose that you want to host two models, A and B, and you assign traffic weight 2 for model A and 1 for model B. Amazon SageMaker distributes two-thirds of the traffic to Model A, and one-third to model B.

## Troubleshooting Deployment
- Use CloudWatch
- Search CloudWatch logs
- Alert on key events
- Use Auto-Scaling
- Enterprise AWS Support (highly recommended - access to AWS experts within minutes)

#### Support Plans
- Basic Support (included for all AWS users)

|Feature| Developer | Business | Enterprise |
|--------|-----------|----------|------------|
||experimenting or testing in AWS| production workloads in AWS|business and/or mission critical workloads in AWS.|
|AWS Trusted Advisor Best Practice Checks|7 Core checks|Full set of checks|Full set of checks|
|Hours| Business hours** email access to Support Engineers| 24x7 phone, email, and chat access to Support Engineers|24x7 phone, email, and chat access to Support Engineers|
|Response time| 12-24 hrs|1-24 hrs|15 min - 24 hrs|
|Guidence| General| Contextual to your use case|Consultative review of your application|
|Pricing| Starts at $29 / month|Starts at $100 / month|Starts at $15K / month|


#### CloudWatch
- Alarms
- Events
- Logs & Insights (can do SQL based queries)

## Production Security 
- Understand KMS Systems (no Encryption by default in SageMaker - ability to enter KMS key)
- IAM Roles for SageMaker (S3 access by default in SageMaker)
- IAM Roles for VPC (no VPC by default in SageMaker)
	- ability to block all access to SageMaker except through VPC
- Ability to connect to GitHub

## Cost Efficiency for ML Systems
- Understand Spot Instances
	- huge savings
	- ability to define a block of time
	- good for batch based and EMR applications
	- can be terminated in 2 minutes notice
	- instance termination can cause data loss (use EBS volume to store data, artifacts, logs)
	- Tip: Decouple compute, storage, and code artifacts & keep compute instance stateless
	- Tip: Use version control with Git
	- Tip: Automate instance replacement after termination
- Undertand proper use of CPU v. GPU resources
- Scale up/down
- Improve time-to-market
- Choose wisely from AI API v. "do it yourself"

>When is it appropriate to train on low-cost CPU versus high-cost GPUs?

---
# Create a Production ML Application

### ML Data Pipelines
Note: There is usually A 5:1 ratio of data engineers to ML engineers due to the complexity and importance of data architecting
- [Anti-pattern] Accidental Pipeline = production SQL server is center
	- chron job may cause outages or performance issues
	- prone to security issues 
- [Pattern] batch Data lake = Batch ML Pipeline to and from a Data Lake (ideal pipeline)
	- metadata is catalogued with AWS glue
	- tables are exported to S3
	- SageMaker uses S3 data to train models
- [Pattern] Batch Infinite Resources on Production = environment issues
	- Production Aurora cluster
	- Data Scientist can work on a read-only copy of the production database

### Modeling in SageMaker (Demo with "countycensusclustering" notebook in Sagemaker
1. Read in data from S3
2. Perform EDA (cleaning & exploration)
	- clean
	- visualize
	- feature engineering
3. Data Modeling
	- dimentionality reduction
	- accessing the PCA model attributes
	- deploying the PCA model
	- population segmentation using unsupervisedclustering
4. Draw Conclusion
	- accessing the kMeans model attributes 

---
# Case Studies

## SageMaker Features

### Search
- Find training jobs
- Rank training jobs
- trace lineage of a model

### Ground Truth
- set up and manage labling jobs
- uses human workers to label
- contract out to the public (Mechanical Turk) or create a private labeling job

### Notebooks
- Jupyter / Jupyter Lab / Terminal
- Lifecycle Configurations (automate notebook provisioning)
- integrates with a GiHub repository

__Jupyter Labs:__ share notebooks with multi-user interface

### Training 
- create algorithms
- buy algorithms on the Marketplace
- hyperparameter tuning 

### Inference
- Compilation jobs (Neo - optimize a model to work on specific hardware) 
- Model packages
- Models
- Endpoint configurations
- Endpoints
- Batch Transform Jobs

### Built-In Algos
- What are all the Algos?
- Where are they located?
- What type of input mode do they accept?
- What file types?
- What instance classes?

## DeepLens Features
- 16 GB Memory / 8 GB RAM
- Ability to train your own algorithms
- Project stream mode : detects bounding boxes
- Debugging mode: look at the live stream (raw/unfiltered)
- Used for object detection, facial recognition, action detection, etc..

## Kinesis Features
* Handles stream-based data
* Processes data in real-time
	- Feed to EMR/SageMaker
* 100's of TB / hr
* Example inputs are:
	- logs
	- financial transactions
	- treaming data

### Data Streams
Allows you to process data with:
* Your own applications 
* AWS Lambda
* Kinesis Forehose
* Kinesis Analytics

### Data FireHose
Create a new delivery stream:
1. Choose a source
	- Direct PUT (use when sending directly or connecting with AWS IoT, CloudWatch Logs, CloudWatch Evens)
	- Kinesis Stream
2. Process Records
	- take source records
	- transform source records (invoke AWS Lambda)
	- Convert record format so AWS Glue can read
3. Select a Destination
	- S3
	- Redshift
	- Elastic Search
	- Splunk (3rd party time-series database)

### Data Analytics
Run Continuous analytics on streaming data from data streams and Kinesis Firehose

### Video Streams
Build applications to process or analyze streaming media

## AWS Flavored Python
### Boto3
Main interface for working with AWS. Any service in AWS can be communicated with Boto
### Pandas
Main data science library for AWS and Python

* Descriptive Stastics
* Plotting

___Example Production Code:___
Take a data frame of text (each row has a body of text) and determine the sentiment for each document (row). 

1. Import logging `!pip -q install python-json-logger`
2. Setup Logging
	```python
	import logging
	from pythonjsonlogger import jsonlogger

	LOG = logging.getLogger()
	LOG.setLevel(logging.DEBUG)
	logHandler = logging.StreamHandler()
	formatter = jsonlogger.JsonFormatter()
	logHandler.setFormatter(formatter)
	LOG.addHandler(logHandler)
	```
3. Write production code to go row-by-row on the dataframe and determine the sentiment of the text using AWS Comprehend API service. 

	```python
	import click
	import boto3
	import pandas as pd

	def create_sentiment(row):
		LOG.info(f"Processing {row}")
		comprehend = boto3.client(service_name='comprehend', region_name='us-east-1')
		payload = comprehend.detect_sentiment(Text=row, LanguageCode='en')
		Log.DEBUG(f"Found Sentiment: {payload}")
		sentiment = payload["Sentiment"]
		return sentiment

	def apply_sentiment(df, column="SentimentRaw"):
		df["Sentiment"] = df[column].apply(create_sentiment)
		return df
	```

## Cloud9 and Building ML Apps
Allows you to get access to entire AWS web ecosystem and serverless ecosystem and control it in a JavaScript browser. Its a development environment in the coud using only a browser. 
1. Set environment type
	- EC2
	- SSH
2. Set instance type
3. Set auto-shutoff time
4. Set VPC and IAM

- has ability to connect with GitHub
- great for developing and testing Lambda functions
- ability to install packages locally










