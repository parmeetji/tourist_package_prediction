
# For data manipulation
import pandas as pd
import sklearn
# For creating a folder
import os
# For data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split
# For hugging face space authentication to upload files
from huggingface_hub import login, HfApi

# Defining constants for the dataset and output paths
api = HfApi(token=os.getenv("HF_TOKEN"))
DATASET_PATH = "hf://datasets/Pammi123/tourism-product-purchase/tourism.csv"
tourism_dataset = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully.")

# Cleaning 'Gender' attribute as it contains some of the 'Female' as 'Fe Male'
tourism_dataset['Gender'] = tourism_dataset['Gender'].replace('Fe Male', 'Female')

# Define the target variable for the classification task
target = 'ProdTaken'

# List of numerical features in the dataset
numeric_features = [
    'DurationOfPitch',            # Duration of the sales pitch delivered to the customer
    'Age',                        # Customer's age
    'NumberOfPersonVisiting',     # Total number of people accompanying the customer on the trip
    'NumberOfFollowups',          # Total number of follow-ups by the salesperson after the sales pitch
    'PreferredPropertyStar',      # Preferred hotel rating by the customer
    'NumberOfTrips',              # Average number of trips the customer takes annually
    'Passport',                   # Whether the customer holds a valid passport (Binary-> 0: No, 1: Yes)
    'PitchSatisfactionScore',     # Score indicating the customer's satisfaction with the sales pitch
    'OwnCar',                     # Whether the customer owns a car (Binary-> 0: No, 1: Yes)
    'NumberOfChildrenVisiting',   # Number of children below age 5 accompanying the customer
    'MonthlyIncome',              # Gross monthly income of the customer
    'CityTier'                    # The city category based on development, population, and living standards (Tier 1 > Tier 2 > Tier 3)
]

# List of categorical features in the dataset
categorical_features = [
    'TypeofContact',              # The method by which the customer was contacted (Company Invited or Self Inquiry)
    'Occupation',                 # Customer's occupation (e.g., Salaried, Freelancer)
    'Gender',                     # Gender of the customer (Male, Female)
    'MaritalStatus',              # Marital status of the customer (Single, Married, Divorced)
    'ProductPitched',             # The type of product pitched to the customer
    'Designation'                 # Customer's designation in their current organization
]

# Defining predictor matrix (X) using selected numeric and categorical features
X = tourism_dataset[numeric_features + categorical_features]

# Defining target variable
y = tourism_dataset[target]


# Splitting the dataset into training and test sets
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y,              # Predictors (X) and target variable (y)
    test_size=0.2,     # 20% of the data is reserved for testing
    random_state=42    # Ensures reproducibility by setting fixed random seed
)

# Saving individual training and testing sets
Xtrain.to_csv("Xtrain.csv",index=False)
Xtest.to_csv("Xtest.csv",index=False)
ytrain.to_csv("ytrain.csv",index=False)
ytest.to_csv("ytest.csv",index=False)

files = ["Xtrain.csv","Xtest.csv","ytrain.csv","ytest.csv"]

# Uploading the resulting train and test datasets back to the Hugging Face data space
for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path.split("/")[-1],  # just the filename
        repo_id="Pammi123/tourism-product-purchase",
        repo_type="dataset",
    )
