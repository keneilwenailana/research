import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Create output directories if they don't exist
os.makedirs('output', exist_ok=True)
os.makedirs('output_images', exist_ok=True)

# Read the CSV file
def load_data(file_path):
    """
    Load and preprocess the data with more detailed feature engineering
    """
    # [Previous load_data function remains exactly the same]
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

def create_and_save_visualization(df, plot_function, filename, title, figsize=(10, 6)):
    """
    Helper function to create and save individual visualizations
    """
    plt.figure(figsize=figsize)
    plot_function(df)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(f'output_images/{filename}.png')
    plt.close()

def create_separate_visualizations(df):
    """
    Create and save individual visualizations
    """
    # 1. Lending Facilities Distribution (Pie Chart)
    def plot_lending_dist(df):
        lending_dist = df['Has Lending Facilities'].value_counts()
        plt.pie(lending_dist, labels=['No Lending', 'Has Lending'],
               autopct='%1.1f%%', colors=['#FF9999', '#66B2FF'])

    create_and_save_visualization(df, plot_lending_dist,
                                'lending_distribution',
                                'Lending Facilities Distribution')

    # 2. Lending Facilities by Entity Size (Bar Plot)
    def plot_lending_by_size(df):
        lending_by_size = df.groupby('Entity Size')['Has Lending Facilities'].mean() * 100
        lending_by_size.plot(kind='bar', color='#66B2FF')
        plt.xlabel('Entity Size')
        plt.ylabel('% with Lending Facilities')

    create_and_save_visualization(df, plot_lending_by_size,
                                'lending_by_size',
                                'Lending Facilities by Entity Size')

    # 3. Lending Facilities by Province (Bar Plot)
    def plot_lending_by_province(df):
        lending_by_province = df.groupby('Province')['Has Lending Facilities'].mean() * 100
        lending_by_province.plot(kind='bar', color='#66B2FF')
        plt.xlabel('Province')
        plt.ylabel('% with Lending Facilities')
        plt.xticks(rotation=45, ha='right')

    create_and_save_visualization(df, plot_lending_by_province,
                                'lending_by_province',
                                'Lending Facilities by Province')

    # 4. Land Size vs Annual Turnover (Scatter Plot)
    def plot_land_vs_turnover(df):
        sns.scatterplot(data=df, x='Land Size Numeric', y='Annual Turnover Numeric',
                       hue='Has Lending Facilities')
        plt.xlabel('Land Size (Hectares)')
        plt.ylabel('Annual Turnover')

    create_and_save_visualization(df, plot_land_vs_turnover,
                                'land_size_vs_turnover',
                                'Land Size vs Annual Turnover')

    # 5. Diversity vs Lending Facilities (Box Plot)
    def plot_diversity_lending(df):
        sns.boxplot(data=df, x='Has Lending Facilities', y='Total Diversity')
        plt.xlabel('Has Lending Facilities')
        plt.ylabel('Total Diversity')

    create_and_save_visualization(df, plot_diversity_lending,
                                'diversity_vs_lending',
                                'Diversity vs Lending Facilities')

    # 6. Bank Distribution (Pie Chart)
    def plot_bank_dist(df):
        bank_dist = df['Bank'].value_counts()
        plt.pie(bank_dist, labels=bank_dist.index, autopct='%1.1f%%')

    create_and_save_visualization(df, plot_bank_dist,
                                'bank_distribution',
                                'Bank Distribution')

    # 7. Correlation Matrix (Heatmap)
    def plot_correlation(df):
        correlation_columns = ['Annual Turnover Numeric', 'Land Size Numeric',
                             'Livestock Diversity', 'Grain Diversity',
                             'Vegetable Diversity', 'Total Diversity']
        correlation_df = df[correlation_columns].corr()
        sns.heatmap(correlation_df, annot=True, cmap='coolwarm')

    create_and_save_visualization(df, plot_correlation,
                                'correlation_matrix',
                                'Correlation Matrix',
                                figsize=(12, 8))

    # 8. Lending Facilities by Land Ownership (Bar Plot)
    def plot_lending_by_ownership(df):
        lending_by_ownership = df.groupby('Land Ownership')['Has Lending Facilities'].mean() * 100
        lending_by_ownership.plot(kind='bar', color='#66B2FF')
        plt.xlabel('Land Ownership')
        plt.ylabel('% with Lending Facilities')

    create_and_save_visualization(df, plot_lending_by_ownership,
                                'lending_by_ownership',
                                'Lending Facilities by Land Ownership')

    # 9. Entity Type Distribution (Pie Chart)
    def plot_entity_type(df):
        entity_type_dist = df['Type of entity'].value_counts()
        plt.pie(entity_type_dist, labels=entity_type_dist.index, autopct='%1.1f%%')

    create_and_save_visualization(df, plot_entity_type,
                                'entity_type_distribution',
                                'Entity Type Distribution')

def main():
    # Load data
    df = load_data('data.csv')

    # Create separate visualizations
    create_separate_visualizations(df)

    # Create detailed statistical summary
    with open('output/detailed_analysis.txt', 'w') as f:
        # [Previous statistical analysis code remains the same]
        f.write("LENDING FACILITIES SUMMARY\n")
        f.write("==========================\n")
        f.write(f"Total Entities: {len(df)}\n")
        f.write(f"Entities with Lending Facilities: {df['Has Lending Facilities'].sum()}\n")
        f.write(f"Percentage with Lending Facilities: {df['Has Lending Facilities'].mean()*100:.2f}%\n\n")

        from scipy import stats

        # Chi-square tests
        contingency_size = pd.crosstab(df['Entity Size'], df['Has Lending Facilities'])
        chi2_size, p_size = stats.chi2_contingency(contingency_size)[:2]
        f.write("CHI-SQUARE TESTS\n")
        f.write("================\n")
        f.write(f"Entity Size vs Lending Facilities:\n")
        f.write(f"Chi-square statistic: {chi2_size:.4f}\n")
        f.write(f"p-value: {p_size:.4f}\n\n")

        categorical_vars = ['Land Ownership', 'Province', 'Type of entity']
        for var in categorical_vars:
            contingency = pd.crosstab(df[var], df['Has Lending Facilities'])
            chi2, p = stats.chi2_contingency(contingency)[:2]
            f.write(f"{var} vs Lending Facilities:\n")
            f.write(f"Chi-square statistic: {chi2:.4f}\n")
            f.write(f"p-value: {p:.4f}\n\n")

    print("Analysis complete. Please check the 'output_images' directory for individual visualizations")
    print("and the 'output' directory for the statistical analysis.")
    print("\nFiles generated:")
    print("1. output_images/")
    print("   - lending_distribution.png")
    print("   - lending_by_size.png")
    print("   - lending_by_province.png")
    print("   - land_size_vs_turnover.png")
    print("   - diversity_vs_lending.png")
    print("   - bank_distribution.png")
    print("   - correlation_matrix.png")
    print("   - lending_by_ownership.png")
    print("   - entity_type_distribution.png")
    print("2. output/detailed_analysis.txt")

if __name__ == '__main__':
    main()
