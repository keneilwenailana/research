import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

if not os.path.exists('coding_images'):
    os.makedirs('coding_images')

df = pd.read_excel("coding_table.xlsx")


# 1. Bar plot for Code frequencies
def plot_code_frequencies():
    plt.figure(figsize=(12, 6))
    df.sort_values('Count', ascending=True).plot(
        kind='barh',
        x='Code',
        y='Count',
        color='skyblue'
    )
    plt.title('Distribution of Codes by Frequency')
    plt.xlabel('Count')
    plt.ylabel('Code')
    plt.tight_layout()
    plt.savefig('coding_images/code_frequencies.png', dpi=300, bbox_inches='tight')
    plt.close()

# 2. Category-wise analysis
def plot_category_analysis():
    plt.figure(figsize=(12, 6))
    category_summary = df.groupby('Category').agg({
        'Count': 'sum',
        'Cases': 'sum'
    })

    category_summary.plot(kind='bar')
    plt.title('Count vs Cases by Category')
    plt.xlabel('Category')
    plt.ylabel('Number')
    plt.legend(title='Metric')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('coding_images/category_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

# 3. Heatmap of % Codes vs % Cases
def create_heatmap():
    plt.figure(figsize=(10, 8))
    heatmap_data = df.pivot_table(
        values=['% Codes', '% Cases'],
        index='Category',
        columns='Code',
        fill_value=0
    )
    sns.heatmap(heatmap_data, annot=True, fmt='.1f', cmap='YlOrRd')
    plt.title('Heatmap of % Codes vs % Cases')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('coding_images/heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()

# 4. Pie chart for category distribution
def plot_category_distribution():
    plt.figure(figsize=(10, 8))
    category_counts = df.groupby('Category')['Count'].sum()
    plt.pie(category_counts, labels=category_counts.index, autopct='%1.1f%%')
    plt.title('Distribution of Codes by Category')
    plt.axis('equal')
    plt.savefig('coding_images/category_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()

# 5. Comparison of Codes vs Cases percentages
def plot_percentage_comparison():
    plt.figure(figsize=(12, 6))
    plt.scatter(df['% Codes'], df['% Cases'], alpha=0.6)
    plt.xlabel('% Codes')
    plt.ylabel('% Cases')
    plt.title('Comparison of % Codes vs % Cases')

    # Add labels for each point
    for i, txt in enumerate(df['Code']):
        plt.annotate(txt, (df['% Codes'].iloc[i], df['% Cases'].iloc[i]),
                    xytext=(5, 5), textcoords='offset points')

    plt.tight_layout()
    plt.savefig('coding_images/percentage_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

# Function to generate all plots
def generate_all_plots():
    plot_code_frequencies()
    plot_category_analysis()
    plot_category_distribution()

# Generate all visualizations
generate_all_plots()
