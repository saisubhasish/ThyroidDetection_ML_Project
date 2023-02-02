# Thyroid Disease Detection
![image](https://user-images.githubusercontent.com/102937478/216236728-dad1775c-37bc-45a5-861e-9e430ea781e3.png)

### Step 1 - Install the requirements

```bash
pip install -r requirements.txt
```

### Step 2 - Run main.py file

```bash
python main.py
```


To download your dataset

```
wget https://raw.githubusercontent.com/saisubhasish/datasets/main/hypothyroid.csv
```

This is changes made in neuro lab


Git commands

If you are starting a project and you want to use git in your project
```
git init
```
Note: This is going to initalize git in your source code.


OR

You can clone exiting github repo
```
git clone <github_url>
```
Note: Clone/ Downlaod github  repo in your system


Add your changes made in file to git stagging are
```
git add file_name
```
Note: You can given file_name to add specific file or use "." to add everything to staging are


Create commits
```
git commit -m "message"
```

```
git push origin main
```
Note: origin--> contains url to your github repo
main--> is your branch name 

To push your changes forcefully.
```
git push origin main -f
```


To pull  changes from github repo
```
git pull origin main
```
Note: origin--> contains url to your github repo
main--> is your branch name

### Deployment link
```
http://ec2-65-2-6-7.ap-south-1.compute.amazonaws.com:8080/](http://ec2-15-206-149-89.ap-south-1.compute.amazonaws.com:8080)
(Link might have expired)
```
### Problem Statement
Thyroid disease is one of the most common disease with endocrine disorder in the human population today. For example hyperthyroidism (over) and hypothyroidism (under), which are relate to release of amount of thyroid hormones the thyroid gland produces and whether it is over active trusted source (when thyroid gland makes too much thyroid hormone) or under active trusted source (when the thyroid gland doesn’t make enough thyroid hormone). We need to identify whether the patient has thyroid or not.

### Solution Proposed 
We need to build a ML model which will be used by hospitals and help the hospital authority to identify if the patient has thyroid or not. If it is a positive case then medical will do further test to know what type of thyroid the person is suffering from and according to that the treatment will be on fast-track. The doctors will start treating the patients. If the result will come negative then the patient will be sent to a junior doctor and the junior doctors by using their own expertise they will decide that if the model has done correct prediction or not. If analysis comes true then the doctor release the patient. By seeing the readings if doctor analysed that there may be a chance of thyroid then patient sent to the senior doctors. 

## Tech Stack Used
1. Python 
2. VS Code 
3. Machine learning algorithms
4. Docker
5. MongoDB
6. Flask

## Infrastructure Required.

1. AWS S3
2. AWS EC2
3. AWS ECR
4. Github Actions
5. Terraform

## How to run?
Before we run the project, make sure that you are having MongoDB in your local system, with Compass since we are using MongoDB for data storage. You also need AWS account to access the service like S3, ECR and EC2 instances.

## Data Collections
![image](https://user-images.githubusercontent.com/102937478/216246951-7c187908-a8b0-4c64-8f37-6549c49e20fa.png)


## Project Archietecture
![image](https://user-images.githubusercontent.com/102937478/216247171-b2b6c281-db84-4327-9aed-b34d9e1aaa6b.png)

## Deployment Archietecture
![image](https://user-images.githubusercontent.com/57321948/193536973-4530fe7d-5509-4609-bfd2-cd702fc82423.png)
![image](https://user-images.githubusercontent.com/102937478/216247497-0d54dd7b-3916-4670-8465-3f1fcf6e0a9a.png)

### Step 1: Clone the repository
```bash
git clone https://github.com/saisubhasish/APS-Sensor-Fault-Detection
```

### Step 2- Create a conda environment after opening the repository

```bash
conda create -n thyroid python=3.8 -y
```

```bash
conda activate thyroid
```

### Step 3 - Install the requirements
```bash
pip install -r requirements.txt
```

### Step 4 - Export the environment variable
```bash
export AWS_ACCESS_KEY_ID=<AWS_ACCESS_KEY_ID>

export AWS_SECRET_ACCESS_KEY=<AWS_SECRET_ACCESS_KEY>

export AWS_DEFAULT_REGION=<AWS_DEFAULT_REGION>

export MONGODB_URL="mongodb+srv://MongoDB:Sai12345@cluster0.i7o85x8.mongodb.net/?retryWrites=true&w=majority"

```

### Step 5 - Run the application server
```bash
python main.py
```


## Run locally

1. Check if the Dockerfile is available in the project directory

2. Build the Docker image
```
docker build --build-arg AWS_ACCESS_KEY_ID=<AWS_ACCESS_KEY_ID> --build-arg AWS_SECRET_ACCESS_KEY=<AWS_SECRET_ACCESS_KEY> --build-arg AWS_DEFAULT_REGION=<AWS_DEFAULT_REGION> --build-arg MONGODB_URL=<MONGODB_URL> . 

```

3. Run the Docker image
```
docker run -d -p 8080:8080 <IMAGE_NAME>
```

To run the project  first execute the below commmand.
MONGO DB URL: 
```
mongodb+srv://MongoDB:Sai12345@cluster0.i7o85x8.mongodb.net/?retryWrites=true&w=majority
```
windows user

```
MONGO_DB_URL=mongodb+srv://MongoDB:Sai12345@cluster0.i7o85x8.mongodb.net/?retryWrites=true&w=majority
```

Linux user

```
mongodb+srv://MongoDB:Sai12345@cluster0.i7o85x8.mongodb.net/?retryWrites=true&w=majority
```

then run 
```
python main.py
```

### To download the dataset 
```
wget https://raw.githubusercontent.com/saisubhasish/datasets/main/hypothyroid.csv
```

### To check and reset git log
```
git log
git reset --soft 6afd
6afd -> last 4 digit of log. 
```

### To add and uplod to git
```
git add filename
we can also use . for all file(Current directory)

git commit -m "Message"
git push origin main
```

### To run jupyter-notebook in vscode
```
 pip install ipykernel
```

### **To create a new environment in vscode** 
```
 1. Select the command prompt as a terminal 
conda create -p venv python==3.87 -y
```

### Create a .env It contains details.
```
MONGO_DB_URL="mongodb://localhost:27017/neurolabDB"
AWS_ACCESS_KEY_ID="aagswdiquyawvdiu"
AWS_SECRET_ACCESS_KEY="sadoiuabnswodihabosdbn"
```
### **To install dockers in aws machine (EC2)**
```
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker ubuntu
newgrp docker
```

**Secrets**
```
AWS_ACCESS_KEY_ID=
AWS_SECRET_ACCESS_KEY=
AWS_REGION=
AWS_ECR_LOGIN_URI=
ECR_REPOSITORY_NAME=
BUCKET_NAME=
MONGO_DB_URL=
```
