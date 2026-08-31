import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Créer le dossier de sortie
output_dir = 'analysis/mcm_freewater_comparison_figures'
os.makedirs(output_dir, exist_ok=True)

# Charger les données
df = pd.read_csv('analysis/mcm_comparison_results.csv')

# Configuration des figures
plt.style.use('default')
sns.set_palette("husl")

# Figure 1: Boxplot des moyennes des poids FreeWater par pipeline
plt.figure(figsize=(12, 8))
sns.boxplot(data=df, x='pipeline', y='pipeline_mean')
plt.xticks(rotation=45, ha='right')
plt.title('Distribution des moyennes des poids FreeWater par pipeline')
plt.xlabel('Pipeline')
plt.ylabel('Moyenne des poids FreeWater')
plt.tight_layout()
plt.savefig(f'{output_dir}/freewater_means_boxplot.png', dpi=300, bbox_inches='tight')
plt.close()

# Figure 2: Scatter plot des moyennes originales vs moyennes pipeline
plt.figure(figsize=(10, 8))
for pipeline in df['pipeline'].unique():
    subset = df[df['pipeline'] == pipeline]
    
    # Calculate IQR and remove outliers
    Q1 = subset['pipeline_mean'].quantile(0.25)
    Q3 = subset['pipeline_mean'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound =subset['original_mean'].min()- 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    subset_filtered = subset[(subset['pipeline_mean'] >= lower_bound) & 
                              (subset['pipeline_mean'] <= upper_bound)]
    
    plt.scatter(subset_filtered['original_mean'], subset_filtered['pipeline_mean'], 
                label=pipeline, alpha=0.7)

plt.plot([0, 1], [0, 1], 'r--', label='Ligne d\'égalité')
plt.xlabel('Moyenne originale')
plt.ylabel('Moyenne pipeline')
plt.title('Comparaison des moyennes des poids FreeWater (sans outliers)')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.xlim(lower_bound, upper_bound)
plt.ylim(lower_bound, upper_bound)
plt.savefig(f'{output_dir}/freewater_means_scatter.png', dpi=300, bbox_inches='tight')
plt.close()

# Figure 3: Heatmap des différences absolues moyennes
pivot_abs_diff = df.pivot(index='subject', columns='pipeline', values='abs_diff_mean')
plt.figure(figsize=(12, 8))
sns.heatmap(pivot_abs_diff, annot=True, fmt='.4f', cmap='YlOrRd')
plt.title('Différences absolues moyennes des poids FreeWater')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(f'{output_dir}/freewater_abs_diff_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()

# Figure 4: Heatmap des corrélations
pivot_corr = df.pivot(index='subject', columns='pipeline', values='correlation')
plt.figure(figsize=(12, 8))
sns.heatmap(pivot_corr, annot=True, fmt='.4f', cmap='RdYlGn', vmin=0.8, vmax=1.0)
plt.title('Corrélations des poids FreeWater')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(f'{output_dir}/freewater_correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()

# Figure 5: Bar plot des différences moyennes par pipeline
plt.figure(figsize=(12, 8))
mean_diff_by_pipeline = df.groupby('pipeline')['diff_mean'].mean().sort_values()
mean_diff_by_pipeline.plot(kind='bar')
plt.title('Différences moyennes des poids FreeWater par pipeline')
plt.xlabel('Pipeline')
plt.ylabel('Différence moyenne')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(f'{output_dir}/freewater_mean_diff_bar.png', dpi=300, bbox_inches='tight')
plt.close()

# Figure 6: Boxplot des différences absolues par pipeline
plt.figure(figsize=(12, 8))
sns.boxplot(data=df, x='pipeline', y='abs_diff_mean')
plt.xticks(rotation=45, ha='right')
plt.title('Distribution des différences absolues moyennes par pipeline')
plt.xlabel('Pipeline')
plt.ylabel('Différence absolue moyenne')
plt.tight_layout()
plt.savefig(f'{output_dir}/freewater_abs_diff_boxplot.png', dpi=300, bbox_inches='tight')
plt.close()

print(f"Figures sauvegardées dans le dossier {output_dir}/")