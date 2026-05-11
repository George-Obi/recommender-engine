"""Recommendation engine that maps tech stacks to likely roles."""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

def load_data(csv_path: str) -> pd.DataFrame:
    df= pd.read_csv(csv_path)
    return df


def prepare_features(df: pd.DataFrame):
    tech_columns = [col for col in df.columns if col != "role"]
    X = df[tech_columns]
    y = df["role"]
    return X, y


def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model= RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train,y_train)
    accuracy= model.score(X_test,y_test)
    print(f'Model Accuracy: {accuracy:.2f}')
    return model


def recommend(role_model, tech_input):
    """Generate a role recommendation for a given tech stack input.

    tech_input can be:
    - dict: {"python": 1, "java": 0, ...} (1 for known, 0 for unknown)
    - list/tuple: ordered values matching the model's feature columns
    """
    if hasattr(role_model, "feature_names_in_"):
        feature_columns = list(role_model.feature_names_in_)
    else:
        raise ValueError("Model does not expose feature names. Ensure it's a trained sklearn model.")

    if isinstance(tech_input, dict):
        # Convert dict to list in the order of feature_columns
        values = [tech_input.get(col, 0) for col in feature_columns]
    elif isinstance(tech_input, (list, tuple)):
        if len(tech_input) != len(feature_columns):
            raise ValueError(f"Expected {len(feature_columns)} features, got {len(tech_input)}")
        values = list(tech_input)
    else:
        raise TypeError("tech_input must be a dict or list/tuple")

    # Ensure values are binary (0 or 1)
    values = [int(bool(v)) for v in values]

    input_df = pd.DataFrame([values], columns=feature_columns)
    prediction = role_model.predict(input_df)
    return prediction[0]



if __name__ == "__main__":
    data = load_data("data/tech_stack_roles_v2.csv")
    X, y = prepare_features(data)
    print("Dataset loaded:", len(X), "samples")
    print("Feature columns:", list(X.columns))
    print("Possible roles:", sorted(y.unique()))
    model = train_model(X, y)
    
    print('======================INPUT TECH STACK=======================')
    print('Enter 0 for missing tech stack and 1 for present tech stack')
    
    python= input('Python: ')
    java= input('Java: ')
    javascript= input('Javascript: ')
    react= input('react: ')
    sql= input('SQL: ')
    aws= input('AWS: ')
    docker=input('Docker:')
    ml= input('ML: ')
    tensorflow= input('Tensorflow: ')
    spark= input('Spark:')
    kubernetes= input('Kubernetes:')
    terraform= input('Terraform:')
    swift= input('Swift: ')
    kotlin= input('Kotlin: ')
    graphql= input('Graphql: ')
    redis= input('Redis: ')
    kafka= input('Kafka: ')
    pytorch= input('Pytorch: ')
    tableau= input('Tableau: ')
    bash= input('Bash: ')
    example_tech_stack=[python,java,javascript,react,sql,aws,docker,ml,tensorflow,spark,kubernetes,
                        terraform,swift,kotlin,graphql,redis,kafka,pytorch,tableau,bash]

    recommended_role = recommend(model, example_tech_stack)
    print(f"recommendation for {example_tech_stack}: {recommended_role}")