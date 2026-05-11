# Tech Stack Role Recommendation Engine

## Project Overview

This is **Project 3** of the DecodeLabs Internship - an intelligent recommendation engine that predicts professional roles based on a developer's technology stack. Built using machine learning, this tool helps match individuals to appropriate career paths by analyzing their technical skills.

## Features

- **Machine Learning Classification**: Uses RandomForestClassifier to predict roles with high accuracy
- **Interactive User Input**: Command-line interface for easy technology stack entry
- **Binary Skill Input**: Simple 0/1 input (0 for unknown, 1 for known tech)
- **Flexible Input Formats**: Supports both dictionary and list-based tech stack inputs
- **Feature Extraction**: Automatic feature column identification from training data

## Technologies Supported

The model recognizes 20 different technologies across various domains:

- **Languages**: Python, Java, JavaScript, Swift, Kotlin, Bash
- **Frontend**: React, GraphQL
- **Backend & DevOps**: Docker, AWS, Kubernetes, Terraform
- **Data**: SQL, Spark, Tableau, Kafka, Redis
- **ML/AI**: ML, TensorFlow, PyTorch

## Installation

### Prerequisites

- Python 3.7+
- Required libraries listed in `requirements.txt`

### Setup

1. **Clone/Download the project**

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Ensure data file exists**
   - Place `tech_stack_roles_v2.csv` in the `data/` directory
   - CSV should contain technology columns and a `role` column

## Usage

### Running the Recommender

Execute the script from the recommender directory:

```bash
python recommender.py
```

### How It Works

1. **Data Loading**: The script loads the training data from `data/tech_stack_roles_v2.csv`
2. **Model Training**: RandomForestClassifier trains on the dataset
3. **User Input**: You'll be prompted to enter your tech stack (0 or 1 for each technology)
4. **Prediction**: The model recommends a role based on your input

### Example Interactive Session

```
Enter 0 for missing tech stack and 1 for present tech stack
Python: 1
Java: 0
Javascript: 1
React: 1
SQL: 1
AWS: 1
Docker: 1
ML: 0
Tensorflow: 0
Spark: 0
Kubernetes: 1
Terraform: 0
Swift: 0
Kotlin: 0
Graphql: 1
Redis: 1
Kafka: 0
Pytorch: 0
Tableau: 0
Bash: 1

recommendation for [tech_stack]: Full Stack Developer
```

## Data Format

The CSV file should have the following structure:

| python | java | javascript | ... | role |
|--------|------|------------|-----|------|
| 1 | 0 | 1 | ... | Full Stack Developer |
| 1 | 1 | 0 | ... | Backend Developer |

- Each technology column contains binary values (0 or 1)
- The `role` column contains the target job role
- Include all 20 supported technologies as columns

## Project Structure

```
recommender/
├── README.md                      # This file
├── requirements.txt               # Project dependencies
├── recommender.py                 # Main recommendation engine
└── data/
    └── tech_stack_roles_v2.csv    # Training dataset
```

## Model Performance

The model prints its accuracy on a test set (80/20 split):

```
Model Accuracy: 0.58
```

## API Functions

### `load_data(csv_path: str) -> pd.DataFrame`
Loads the CSV training data.

### `prepare_features(df: pd.DataFrame)`
Extracts features (X) and target (y) from the dataset.

### `train_model(X, y)`
Trains the RandomForestClassifier and returns the trained model.

### `recommend(role_model, tech_input)`
Generates a role recommendation for a given tech stack.
- **Parameters**:
  - `role_model`: Trained RandomForestClassifier
  - `tech_input`: Dict or list of binary technology values
- **Returns**: Recommended role (string)

## Requirements

See `requirements.txt` for all dependencies:
- pandas
- scikit-learn

## Notes

- Model accuracy depends on training data quality and diversity
- Binary input format ensures consistent predictions
- Feature column order is automatically determined from training data

## Author

RAMZY

## License

Internal use only
