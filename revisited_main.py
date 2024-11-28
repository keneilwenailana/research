import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from scipy import stats
from sklearn.preprocessing import LabelEncoder

# Create output directory if it doesn't exist
os.makedirs('output', exist_ok=True)

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

def create_advanced_visualizations(df):
    """
    Create advanced visualizations to explore lending facilities
    """
    plt.figure(figsize=(20, 15))

    # 1. Correlation Heatmap with Lending Facilities
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
        'Bank Encoded'
    ]

    # Compute correlation matrix
    correlation_matrix = df[correlation_columns].corr()

    plt.subplot(2, 2, 1)
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('Correlation Heatmap with Lending Facilities')
    plt.tight_layout()

    # 2. Feature Importance for Lending Facilities
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler

    # Prepare features for importance analysis
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

    # Prepare X and y
    X = df[features]
    y = df['Has Lending Facilities']

    # clean
    merged = pd.concat([X, y], axis=1)
    merged = merged.dropna()
    X = merged.iloc[:, :X.shape[1]]  # separate X rows and columns
    y = merged.iloc[:, X.shape[1]:]

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Random Forest for feature importance
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_scaled, y)

    # Plot feature importances
    plt.subplot(2, 2, 2)
    feature_importance = pd.Series(rf.feature_importances_, index=features)
    feature_importance.sort_values(ascending=True).plot(kind='barh')
    plt.title('Feature Importance for Lending Facilities')
    plt.xlabel('Importance Score')

    # 3. Logistic Regression Coefficients
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Logistic Regression
    lr = LogisticRegression(random_state=42)
    lr.fit(X_train, y_train)

    # Plot logistic regression coefficients
    plt.subplot(2, 2, 3)
    coef_series = pd.Series(lr.coef_[0], index=features)
    coef_series.sort_values(ascending=True).plot(kind='barh')
    plt.title('Logistic Regression Coefficients')
    plt.xlabel('Coefficient Value')

    # 4. Detailed Lending Facilities Distribution
    plt.subplot(2, 2, 4)
    print(df)
    df.groupby('Entity Type Encoded')['Has Lending Facilities'].mean().plot(kind='bar')
    plt.title('Lending Facilities by Entity Type')
    plt.xlabel('Entity Type')
    plt.ylabel('Proportion with Lending Facilities')
    plt.xticks(rotation=45, ha='right')

    # Save the figure
    plt.tight_layout()
    plt.savefig('output/lending_facilities_analysis.png')
    plt.close()

    # Detailed Statistical Analysis
    with open('output/statistical_analysis.txt', 'w') as f:
        # Chi-square tests
        categorical_vars = ['Land Ownership', 'Province', 'Type of entity']
        f.write("CHI-SQUARE INDEPENDENCE TESTS\n")
        f.write("=============================\n")
        for var in categorical_vars:
            try:
                contingency = pd.crosstab(df[var], df['Has Lending Facilities'])
                chi2, p_value = stats.chi2_contingency(contingency)[:2]
                f.write(f"{var} vs Lending Facilities:\n")
                f.write(f"Chi-square statistic: {chi2:.4f}\n")
                f.write(f"p-value: {p_value:.4f}\n\n")
            except:
                pass

        # Detailed lending facilities summary
        f.write("LENDING FACILITIES SUMMARY\n")
        f.write("==========================\n")
        f.write(f"Total Entities: {len(df)}\n")
        f.write(f"Entities with Lending Facilities: {df['Has Lending Facilities'].sum()}\n")
        f.write(f"Percentage with Lending Facilities: {df['Has Lending Facilities'].mean()*100:.2f}%\n\n")

        # Detailed feature statistics
        f.write("FEATURE STATISTICS BY LENDING FACILITIES\n")
        f.write("=======================================\n")
        numeric_features = [
            'Annual Turnover Numeric',
            'Land Size Numeric',
            'Total Diversity'
        ]
        for feature in numeric_features:
            f.write(f"{feature} Statistics:\n")
            f.write("With Lending Facilities:\n")
            f.write(df[df['Has Lending Facilities']][feature].describe().to_string())
            f.write("\n\nWithout Lending Facilities:\n")
            f.write(df[~df['Has Lending Facilities']][feature].describe().to_string())
            f.write("\n\n")

def main():
    # Load data
    df = load_and_preprocess_data('data.csv')

    # Create advanced visualizations
    create_advanced_visualizations(df)

    print("Analysis complete. Please check the 'output' directory for results.")
    print("Files generated:")
    print("1. lending_facilities_analysis.png - Comprehensive analysis visualizations")
    print("2. statistical_analysis.txt - Detailed statistical analysis")

if __name__ == '__main__':
    main()
