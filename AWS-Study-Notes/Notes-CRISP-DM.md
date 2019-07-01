# How to Manage a Data Science Project (CRISP-DM on AWS)

### Cross Industry Standard Process - Data Mining
Used as a framework for your data science projects
(Developed in 1996)

## The Methodology
The 6 Phases (used in a cyclical Process) of Data Science:
1. Business Understanding
2. Data Understanding
3. Data Preparation
4. Modeling
5. Evaluation
6. Deployment

More insights leads to better business understanding which kicks off the process all over again.


## Phase 1: Business Understanding
Start with a clear user story and understanding before getting lost in the data. 

___Is the problem suitable for ML or not?___ 

If it is, then we can process to the next phase: Data Understanding.

### 1. Understant business requirements
Identify the busniness ares that need to improve and convert that to a problem that needs to be solved or a question that needs to be answered. 

Formualte a problem that is both ___specific___ and ___measurable___ is of the most importance to the success of the project. Be sure to highlight the project's critical features including people and resources. 

### 2. Analyze supporting information
- List all the required resources and assumptions
- Understan all associated riskd like business risks and financial contraints
- Make a contigency plan 
- Compare costs and benefits

### 3. Convert to ML Objective
Convert the business question to an ML objective.
- Create a technical data science objective
- Identify the algorithms to be tested and used
- Define the success criteria

### 4. Create a Project Plan

- Number and duration of stages (timeline / roadmap)
- Dependencies
- Risks
- Business and Data Mining Goals
- Evaluation methods
- Tools and Techniques for each stage

## Phase 2: Data Understanding 

### Data Collection
- Identify which data you will be using for the project at hand
	- How it is housed?
	- How you will aquire it?
- Analyze data for additional requirements
	- Do I select _all_ the data or only specific fields
	- Are there missing values?
	- Does the data need to be encoded or decoded?
- Consider other data sources
	

### Data Properties
Look for interesting patterns in the data and use basic stastical techniques to undersand the properties of the data set.
- Describe the data
	- Structured / unstructured
	- Size / volume of data
	- Relationships & complexity
- Identify key features and relationships
	- Mean, median, skewness (basic stastics)
	- List the meaning of these in both technical and business terms
	- Determine main attributes and look for correlations
	- Choose which attributes you want to keep for the model
- Use tools & techniques to explore data properties
	- Spark / SQL / Python
	- Visualizations, graphs, queries, reporting

### Data Quality
Understand the quality of the dataset and whether or not it is ready to be used.

___Before moving forward, write down the likely paths forward for the project.___

- Verify Attributes

- Identify Missing Data: How to handle the missing values?
	- Use an algorithm to predict the value?
	- Drop them?
	- Impute them?
- Reveal inconsistencies: How to handle inconsistent data?
- Report Solution

### AWS Services for Data Understanding
- Athena (query)
- Quicksight (display / visualize)
- AWS GLue (transform)

#### AWS Glue
An ETl service that helps developers with the 'heavy lifting' of data transformation in production. 

Glue does the following:
1. Creates a Data Catalog 
2. Generates & edits transformations (development platform)
3. Schedules & runs jobs (serverless execution)

#### Athena
Serverless interactive query service capable of querying data stored in S3 Data Lake. 
- Infers the schema upon read
- Supports ANSI SQL operations & functions
- Serverless
- Easy to use
- Use with ETL to transform into columnar formats

#### QuickSight
Fast analytics BI dashboard tool. 

- Cloud powered BI tool
- Scale to hundreds/thousands of users
- 1/10th the cost of others
- Secure sharing and collaboration (StoryBoard)

## Phase 3 & 4: Data Preparation & Modeling
These two phases go hand-in-hand and are iterative.

### Data Selection 
- Final dataset selection 
	- Keep in mind the constraints around:
		- Total size?
		- Attributes (which columns)
		- Record Selection (which rows)
		- Data Type
### Data Preparation
- Clean:
	- How is missing data handled?
		- dropping records (row)
		- Add a default value 
		- Use stastical methods
	- Clean up corrupt and variable data
		- look for outliers
		- look for invalid values
- Transform:
	- Derive additional attributes for original attributes
	- Normalize (scale)
	- Attribute transformation (i.e. one hot encode)
- Merge tables to form one DataFrame 
	- Keep an eye out for records that get dropped in this process
	- Revisit Data Understanding phase afterwards
- Format:Make formatting changes to the dataset to accomadate your modeling tools needs. 
	- Rearrange attributes
	- Randomly shuffle
	- Remove constraints of the modeling tool 
		- Remove Unicode characters
		- Change VARCHAR to integers

### Modeling
This phase works together with the Data Preparation phase and is an iterative process. 

- __Model Selection__
	- Modeling Technique Examples:
		- Regression for numeric prediction 
		- Random Forest for Multi-class Calssification
		- RNNs for sequence prediction
	- Consider constraints of modelling technique and tool.
		- Classical Models = SciKit Learn
		- Neural Nets = MXNet, Keras, tensorFlow
	- Think about the ways that the constarints tie back to the Data Preparation phase. 
- __Model Testing Plan__
	- How will you test model efectiveness? What is your Train/Test technique?
	- How will you split the data?
	- How will you evaluate?
		- MSE
		- Precision / Accuracy / Recall
- __Parameter Tuning__
	- Train the model
	- Tweak the model for better performance
	- Build multiple models with different parameter settings
	- Describe the trained models and report the findings. 
		- Why did you choose a model and why did you choose the parameters you did?

### Apache Spark on Amazon EMR
- Use EMR with Spark MLLib DataFrame based API for ML
- Use in IPython notebooks (other notebooks supported as well)
- Scala, Python, R, Java, SQL supported
- Leverage Spot instances for task nodes for cost savings
	- Python AWS Deep Learning AMI with pre-installs:
		- GPU CUDA support for training
		- Deep Learning Frameworks: MXNet, TensorFlow, 
Caffe2, Torch, Keras, Theano
		- Anaconda Data Science Platform with Numpy, SkLearn, etc...
	- Install R Studio on Amazon EC2

___If the model needs more or less data, go back to the Data Preparation phase and start over again. Otherwise, move on to the next phase: Evaluation.___


## Phase 5: Evaluation

- Model evaluation: How does each model perform in relation to the business goals?
	- Accuracy:
		- Use the evaluation metrics that are built into the chosen framework used to build the model
		- Use visualization techniques in validation with libraries like Seaborn
 	- Model generalization on unseen/unknown data
	- Evaluation of the model using business success criteria
	- Assess the steps taken in each phase: Was important criteria overlooked?
	
- Final decision: Deploy or not to deploy?
	- Perform QA checks:
		- How does the model perform using determined criteria?
		- Is the data available for future training?
	- Determine next steps: Launch, iterate, or start anew?
		- Approach the same problem from a differnt angle?
		- Re-define the business goals and start again?
- Store Model in S3 and terminate any running instances to reduce costs. 

Decide what outcomes you are expecting from the model. Are there false positives or False Negatives? What do they mean and what are the business tradeoffs? 

___If the outcome of the model aligns with the original business objectives, then you can move the project forward with Phase 6: Deployment.___

## Phase 6: Deployment (Dev Ops)

### AWS Deployment Services 

#### Runtime
- EC2: cloud-based virtual servers managed by you
- Lambda: Serverless
- EC2 Container Service + Batch: Help run your model on virtual servers in cloud via Docker containers

### Application Deployment
- CodeDeploy: 
	- Automate model and software deployment
	- Integrates with many developer tools
	- Deploys to EC2
	- Requires application infrasture set-up separately
- OpsWorks:
	- Designed for configuartion management
	- Chef: Treats server configuration as code
	- Automate how servers are configured, deployed, and managed across EC2 or on-prem environments
- Elastic Beanstalk:
	- Runs models on virtual servers in the coloud
	- Streamlined deployment & management
	- Good for developers who want to build 2-3 tier level applications without worrying about infrastructure
	- Simply give it a code package and a few settings and the rest is managed by AWS

#### Infrastructure Deployment
- OpsWorks:
	- Deploy & manage __both__ applications + infrastructure
	- deploy resourcces into "stacks"
- Elastic Beanstalk:
	- Handles management & deployment
	- Limits deployment to EC2 & elastic load balancing
	- While it can integrate with AWS services, it does not actively manage those services
- CloudFormation:
	- Set up application infrastructure using streamlined templates in JSON format
	- Easy to reproduce environments

#### Code Management
- CodeCommit:
	- Fully managed source control service
	- integrates with Git tools
- CodePipeline:
	- Continuous Deployment 
	- Continuous Delivery 
	- Used to easily re-deploy new versions/releases of your application
- Elastic Beanstalk

### Maintenance & Monitoring
- CloudWatch
	- Monitor health & performance of applications
	- Set up alarms and notifications
- CloudTrail
	- Identify all API activity among AWS services
	- Find cause of problems faster
- Elastic Beanstalk:
	- Built-in monitoring
	- Trigger server replacement
	- Auto-scale

### Final Report
- Highlight processes used in the project
- Analyze all the goals of the project (were they met?)
- Detail the findings
- Identify & explain the model and your reason for using that particular model
- Identify customer groups to target using this model

### Project Review
- Assess the outcomes of the project
- Summarize the results and write thorough documentation.
	- What went right?
	- What went wrong?
	- What can be improved?
- Generalize the whole process to make it useful for the next iteration. 
	- Which parts can be reused to build out future projects?
	- Which parts need re-working?





