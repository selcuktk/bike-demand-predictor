import matplotlib.pyplot as plt

models = [
    "Starter Model", "Model 02", "Model 03", 
    "Model 04", "Model 05", "Model 06", "Model 07"
]

composite_scores = [24.42, 24.85, 33.37, 24.23, 22.43, 22.12, 29.62]

plt.figure(figsize=(10, 6))
bars = plt.bar(models, composite_scores, color='skyblue', edgecolor='black')

plt.xlabel('Models')
plt.ylabel('Composite Score')
plt.title('Composite Score Comparison of Models')
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)

for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.2f}', ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.show()