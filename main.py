import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Create output directory if it doesn't exist
os.makedirs('output', exist_ok=True)

# Read the CSV file
def load_data(file_path):
    """
    Load and preprocess the data with more detailed feature engineering
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

    # Detailed Entity Size Categorization
    df['Annual Turnover Numeric'] = df['Annual Turnover'].map({
        '<R500k': 250000,
        'R500k - R1m': 750000,
        'R1m - R1.5m': 1250000,
        'R2m - R3m': 2500000
    })

    # Entity Size with more granularity
    df['Entity Size'] = pd.cut(
        df['Annual Turnover Numeric'],
        bins=[-float('inf'), 500000, 1000000, float('inf')],
        labels=['Small', 'Medium', 'Large']
    )

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

    # Land Ownership
    df['Land Ownership'] = df['Does the entity own the land?'].map({
        'Yes': 'Owned',
        'No': 'Not Owned'
    })

    # Bank Categorization
    df['Bank'] = df['Bank'].fillna('No Bank')

    # Province Categorization
    df['Province'] = df['Location/Province'].fillna('Unknown')

    return df

def create_comprehensive_visualizations(df):
    """
    Create a comprehensive set of visualizations
    """
    # Set up the plot style

    # Create multiple figure layouts
    fig, axes = plt.subplots(3, 3, figsize=(20, 20))
    fig.suptitle('Comprehensive Agricultural Lending Facilities Analysis', fontsize=16)

    # 1. Pie Chart: Lending Facilities Distribution
    lending_dist = df['Has Lending Facilities'].value_counts()
    axes[0, 0].pie(lending_dist, labels=['No Lending', 'Has Lending'],
                   autopct='%1.1f%%', colors=['#FF9999', '#66B2FF'])
    axes[0, 0].set_title('Lending Facilities Distribution')

    # 2. Bar Plot: Lending Facilities by Entity Size
    lending_by_size = df.groupby('Entity Size')['Has Lending Facilities'].mean() * 100
    lending_by_size.plot(kind='bar', ax=axes[0, 1], color='#66B2FF')
    axes[0, 1].set_title('Lending Facilities by Entity Size')
    axes[0, 1].set_xlabel('Entity Size')
    axes[0, 1].set_ylabel('% with Lending Facilities')

    # 3. Bar Plot: Lending Facilities by Province
    lending_by_province = df.groupby('Province')['Has Lending Facilities'].mean() * 100
    lending_by_province.plot(kind='bar', ax=axes[0, 2], color='#66B2FF')
    axes[0, 2].set_title('Lending Facilities by Province')
    axes[0, 2].set_xlabel('Province')
    axes[0, 2].set_ylabel('% with Lending Facilities')
    plt.setp(axes[0, 2].get_xticklabels(), rotation=45, ha='right')

    # 4. Scatter Plot: Land Size vs Annual Turnover
    sns.scatterplot(data=df, x='Land Size Numeric', y='Annual Turnover Numeric',
                    hue='Has Lending Facilities', ax=axes[1, 0])
    axes[1, 0].set_title('Land Size vs Annual Turnover')
    axes[1, 0].set_xlabel('Land Size (Hectares)')
    axes[1, 0].set_ylabel('Annual Turnover')

    # 5. Box Plot: Total Diversity and Lending Facilities
    sns.boxplot(data=df, x='Has Lending Facilities', y='Total Diversity', ax=axes[1, 1])
    axes[1, 1].set_title('Diversity vs Lending Facilities')
    axes[1, 1].set_xlabel('Has Lending Facilities')
    axes[1, 1].set_ylabel('Total Diversity')

    # 6. Pie Chart: Bank Distribution
    bank_dist = df['Bank'].value_counts()
    axes[1, 2].pie(bank_dist, labels=bank_dist.index, autopct='%1.1f%%')
    axes[1, 2].set_title('Bank Distribution')

    # 7. Correlation Heatmap
    correlation_columns = ['Annual Turnover Numeric', 'Land Size Numeric',
                           'Livestock Diversity', 'Grain Diversity',
                           'Vegetable Diversity', 'Total Diversity']
    correlation_df = df[correlation_columns].corr()
    sns.heatmap(correlation_df, annot=True, cmap='coolwarm', ax=axes[2, 0])
    axes[2, 0].set_title('Correlation Matrix')

    # 8. Bar Plot: Lending Facilities by Land Ownership
    lending_by_ownership = df.groupby('Land Ownership')['Has Lending Facilities'].mean() * 100
    lending_by_ownership.plot(kind='bar', ax=axes[2, 1], color='#66B2FF')
    axes[2, 1].set_title('Lending Facilities by Land Ownership')
    axes[2, 1].set_xlabel('Land Ownership')
    axes[2, 1].set_ylabel('% with Lending Facilities')

    # 9. Pie Chart: Entity Type Distribution
    entity_type_dist = df['Type of entity'].value_counts()
    axes[2, 2].pie(entity_type_dist, labels=entity_type_dist.index, autopct='%1.1f%%')
    axes[2, 2].set_title('Entity Type Distribution')

    # Adjust layout and save
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('output/comprehensive_lending_analysis.png')
    plt.close()

    # Additional Detailed Analysis
    def detailed_statistical_summary():
        with open('output/detailed_analysis.txt', 'w') as f:
            # Lending Facilities Summary
            f.write("LENDING FACILITIES SUMMARY\n")
            f.write("==========================\n")
            f.write(f"Total Entities: {len(df)}\n")
            f.write(f"Entities with Lending Facilities: {df['Has Lending Facilities'].sum()}\n")
            f.write(f"Percentage with Lending Facilities: {df['Has Lending Facilities'].mean()*100:.2f}%\n\n")

            # Statistical Tests
            from scipy import stats

            # Chi-square test for independence
            contingency_size = pd.crosstab(df['Entity Size'], df['Has Lending Facilities'])
            chi2_size, p_size = stats.chi2_contingency(contingency_size)[:2]
            f.write("CHI-SQUARE TESTS\n")
            f.write("================\n")
            f.write(f"Entity Size vs Lending Facilities:\n")
            f.write(f"Chi-square statistic: {chi2_size:.4f}\n")
            f.write(f"p-value: {p_size:.4f}\n\n")

            # Similar tests for other categorical variables
            categorical_vars = ['Land Ownership', 'Province', 'Type of entity']
            for var in categorical_vars:
                contingency = pd.crosstab(df[var], df['Has Lending Facilities'])
                chi2, p = stats.chi2_contingency(contingency)[:2]
                f.write(f"{var} vs Lending Facilities:\n")
                f.write(f"Chi-square statistic: {chi2:.4f}\n")
                f.write(f"p-value: {p:.4f}\n\n")

    detailed_statistical_summary()

def main():
    # Load data
    df = load_data('data.csv')

    # Create comprehensive visualizations
    create_comprehensive_visualizations(df)

    print("Analysis complete. Please check the 'output' directory for results.")
    print("Files generated:")
    print("1. comprehensive_lending_analysis.png - Comprehensive visualizations")
    print("2. detailed_analysis.txt - Statistical summary and tests")

if __name__ == '__main__':
    main()
