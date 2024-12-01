import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import os

def load_and_preprocess_data(file_path):
    """
    Load and preprocess the data with enhanced feature engineering
    """
    # Read the CSV file
    df = pd.read_csv(file_path)

    # Clean column names
    df.columns = df.columns.str.strip()

    # Convert 'Has Lending Facilities' to boolean
    df['Has Lending Facilities'] = df['Does the entity have any lending facilities with a financial institution'].map({
        'Yes': True,
        'No': False
    })

    # Water rights
    df['Has water rights'] = df['Does the entity have water rights?'].map({
        'Yes':True,
        'No':False
    })

    # Numeric Encoding
    le = LabelEncoder()

    # Detailed Entity Size Categorization
    df['Annual Turnover Numeric'] = df['Annual Turnover'].map({
        '<R500k': 250000,
        'R500k - R1m': 750000,
        'R1m - R1.5m': 1250000,
        'R2m - R3m': 2500000
    })

    # Comprehensive Diversity Calculation
    def count_diversity(column):
        return len(str(column).split(';')) if pd.notna(column) and column != '' else 0

    df['Livestock Diversity'] = df['Livestock'].apply(count_diversity)
    df['Grain Diversity'] = df['Grain'].apply(count_diversity)
    df['Vegetable Diversity'] = df['Vegetables'].apply(count_diversity)
    df['Total Diversity'] = df['Livestock Diversity'] + df['Grain Diversity'] + df['Vegetable Diversity']

    # Land Size Numeric
    df['Land Size Numeric'] = df['How many hectares is the land?'].map({
        '0 - 10 ha': 5,
        '10 - 20 ha': 15,
        '50 - 100 ha': 75,
        '> 100 ha': 150
    })

    # Encode categorical variables
    df['Land Ownership Encoded'] = le.fit_transform(df['Does the entity own the land?'])
    df['Province Encoded'] = le.fit_transform(df['Location/Province'])
    df['Entity Type Encoded'] = le.fit_transform(df['Type of entity'])
    df['Bank Encoded'] = le.fit_transform(df['Bank'])

    return df

def create_correlation_heatmap(df, output_path='revisited_images/correlation_heatmap.png'):
    """Creates and saves a correlation heatmap."""
    correlation_columns = [
        'Has Lending Facilities',
        'Annual Turnover Numeric',
        'Land Size Numeric',
        'Livestock Diversity',
        'Grain Diversity',
        'Vegetable Diversity',
        'Total Diversity',
        'Land Ownership Encoded',
        'Province Encoded',
        'Entity Type Encoded',
        'Bank Encoded',
        'Has water rights'
    ]

    plt.figure(figsize=(12, 10))
    correlation_matrix = df[correlation_columns].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('Correlation Heatmap with Lending Facilities')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def create_feature_importance_plot(df, output_path='revisited_images/feature_importance.png'):
    """Creates and saves a feature importance plot."""
    features = [
        'Annual Turnover Numeric',
        'Land Size Numeric',
        'Livestock Diversity',
        'Grain Diversity',
        'Vegetable Diversity',
        'Land Ownership Encoded',
        'Province Encoded',
        'Entity Type Encoded'
    ]

    X = df[features]
    y = df['Has Lending Facilities']

    merged = pd.concat([X, y], axis=1).dropna()
    X = merged.iloc[:, :X.shape[1]]
    y = merged.iloc[:, X.shape[1]:]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_scaled, y)

    plt.figure(figsize=(10, 6))
    feature_importance = pd.Series(rf.feature_importances_, index=features)
    feature_importance.sort_values(ascending=True).plot(kind='barh')
    plt.title('Feature Importance for Lending Facilities')
    plt.xlabel('Importance Score')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def create_logistic_regression_plot(df, output_path='revisited_images/logistic_regression.png'):
    """Creates and saves a logistic regression coefficients plot."""
    features = [
        'Annual Turnover Numeric',
        'Land Size Numeric',
        'Livestock Diversity',
        'Grain Diversity',
        'Vegetable Diversity',
        'Land Ownership Encoded',
        'Province Encoded',
        'Entity Type Encoded'
    ]

    X = df[features]
    y = df['Has Lending Facilities']

    merged = pd.concat([X, y], axis=1).dropna()
    X = merged.iloc[:, :X.shape[1]]
    y = merged.iloc[:, X.shape[1]:]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    lr = LogisticRegression(random_state=42)
    lr.fit(X_train, y_train)

    plt.figure(figsize=(10, 6))
    coef_series = pd.Series(lr.coef_[0], index=features)
    coef_series.sort_values(ascending=True).plot(kind='barh')
    plt.title('Logistic Regression Coefficients')
    plt.xlabel('Coefficient Value')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def create_entity_type_distribution(df, output_path='revisited_images/entity_type_distribution.png'):
    """Creates and saves an entity type distribution plot."""
    plt.figure(figsize=(10, 6))
    df.groupby('Entity Type Encoded')['Has Lending Facilities'].mean().plot(kind='bar')
    plt.title('Lending Facilities by Entity Type')
    plt.xlabel('Entity Type')
    plt.ylabel('Proportion with Lending Facilities')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def main():
    # Create output directory
    os.makedirs('revisited_images', exist_ok=True)

    # Load and preprocess data (using your existing load_and_preprocess_data function)
    df = load_and_preprocess_data('data.csv')

    # Create individual visualizations
    create_correlation_heatmap(df)
    create_feature_importance_plot(df)
    create_logistic_regression_plot(df)
    create_entity_type_distribution(df)

    print("Visualizations have been saved in the 'revisited_images' folder:")
    print("1. correlation_heatmap.png")
    print("2. feature_importance.png")
    print("3. logistic_regression.png")
    print("4. entity_type_distribution.png")

if __name__ == '__main__':
    main()
