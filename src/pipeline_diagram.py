# Pipeline diyagramları oluşturma
# Metodoloji ve pipeline görselleştirmeleri

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import src.config as config


def create_methodology_diagram(save_path):
    # Genel metodoloji diyagramını oluştur
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis('off')
    
    # Define stages
    stages = [
        ("Data Collection", 1, 10),
        ("Data Preprocessing", 3, 10),
        ("Feature Engineering", 5, 10),
        ("Spatial Analysis", 7, 10),
        ("Model Selection", 1, 7),
        ("Model Training", 3, 7),
        ("Hyperparameter Tuning", 5, 7),
        ("Model Evaluation", 7, 7),
        ("Interpretability Analysis", 1, 4),
        ("Deployment", 3, 4),
        ("Monitoring and Updating", 5, 4),
    ]
    
    # Draw boxes
    boxes = []
    for stage, x, y in stages:
        box = FancyBboxPatch(
            (x-0.4, y-0.3), 0.8, 0.6,
            boxstyle="round,pad=0.05",
            edgecolor='black',
            facecolor='lightblue',
            linewidth=1.5
        )
        ax.add_patch(box)
        ax.text(x, y, stage, ha='center', va='center', fontsize=9, weight='bold')
        boxes.append((x, y))
    
    # Draw arrows
    arrows = [
        (1, 10, 3, 10),  # Collection -> Preprocessing
        (3, 10, 5, 10),  # Preprocessing -> Feature Engineering
        (5, 10, 7, 10),  # Feature Engineering -> Spatial Analysis
        (7, 10, 7, 7.3),  # Spatial Analysis -> Evaluation
        (7, 7, 5, 7),    # Evaluation -> Tuning
        (5, 7, 3, 7),    # Tuning -> Training
        (3, 7, 1, 7),    # Training -> Selection
        (1, 7, 1, 4.3),  # Selection -> Interpretability
        (1, 4, 3, 4),    # Interpretability -> Deployment
        (3, 4, 5, 4),    # Deployment -> Monitoring
    ]
    
    for x1, y1, x2, y2 in arrows:
        arrow = FancyArrowPatch(
            (x1, y1-0.3 if y1 > y2 else y1+0.3),
            (x2, y2+0.3 if y1 > y2 else y2-0.3),
            arrowstyle='->',
            mutation_scale=20,
            linewidth=2,
            color='darkblue'
        )
        ax.add_patch(arrow)
    
    ax.set_title('Methodology of Proposed Model', fontsize=16, weight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(save_path, dpi=config.VISUALIZATION['dpi'], bbox_inches='tight')
    plt.close()
    print(f"Saved methodology diagram to: {save_path}")


def create_project_pipeline_diagram(save_path):
    """Create project-specific pipeline diagram"""
    fig, ax = plt.subplots(figsize=(18, 12))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 14)
    ax.axis('off')
    
    # Define stages specific to this project
    stages = [
        ("CSV Data\nLoading", 1, 12),
        ("Column\nMapping", 3, 12),
        ("Missing Value\nAnalysis", 5, 12),
        ("Data Cleaning\n(Outliers)", 7, 12),
        ("Target\nTransformation", 9, 12),
        ("Feature\nEngineering", 1, 9),
        ("EDA\nVisualizations", 3, 9),
        ("Volatility\nAnalysis", 5, 9),
        ("Clustering\nAnalysis", 7, 9),
        ("Train/Test\nSplit", 9, 9),
        ("Model Training\n(RF + XGB)", 1, 6),
        ("Hyperparameter\nTuning", 3, 6),
        ("Model\nEvaluation", 5, 6),
        ("Feature\nImportance", 7, 6),
        ("Anomaly\nDetection", 9, 6),
        ("Report\nGeneration", 5, 3),
    ]
    
    # Color coding
    colors = {
        'data': 'lightblue',
        'analysis': 'lightgreen',
        'modeling': 'lightyellow',
        'output': 'lightcoral'
    }
    
    # Draw boxes
    for stage, x, y in stages:
        if 'Loading' in stage or 'Mapping' in stage or 'Missing' in stage or 'Cleaning' in stage or 'Transformation' in stage:
            color = colors['data']
        elif 'EDA' in stage or 'Volatility' in stage or 'Clustering' in stage:
            color = colors['analysis']
        elif 'Training' in stage or 'Tuning' in stage or 'Evaluation' in stage or 'Importance' in stage or 'Split' in stage:
            color = colors['modeling']
        else:
            color = colors['output']
        
        box = FancyBboxPatch(
            (x-0.5, y-0.4), 1.0, 0.8,
            boxstyle="round,pad=0.05",
            edgecolor='black',
            facecolor=color,
            linewidth=1.5
        )
        ax.add_patch(box)
        ax.text(x, y, stage, ha='center', va='center', fontsize=8, weight='bold')
    
    # Draw arrows (simplified flow)
    arrows = [
        (1, 12, 3, 12), (3, 12, 5, 12), (5, 12, 7, 12), (7, 12, 9, 12),
        (9, 12, 9, 9.4), (9, 9, 7, 9), (7, 9, 5, 9), (5, 9, 3, 9), (3, 9, 1, 9),
        (1, 9, 1, 6.4), (1, 6, 3, 6), (3, 6, 5, 6), (5, 6, 7, 6), (7, 6, 9, 6),
        (1, 6, 5, 3.4), (3, 6, 5, 3.4), (5, 6, 5, 3.4), (7, 6, 5, 3.4), (9, 6, 5, 3.4),
        (5, 9, 5, 3.4),  # Analysis to report
    ]
    
    for x1, y1, x2, y2 in arrows:
        arrow = FancyArrowPatch(
            (x1, y1-0.4 if y1 > y2 else y1+0.4),
            (x2, y2+0.4 if y1 > y2 else y2-0.4),
            arrowstyle='->',
            mutation_scale=15,
            linewidth=1.5,
            color='darkblue',
            alpha=0.7
        )
        ax.add_patch(arrow)
    
    # Legend
    legend_elements = [
        mpatches.Patch(facecolor=colors['data'], label='Data Processing'),
        mpatches.Patch(facecolor=colors['analysis'], label='Analysis'),
        mpatches.Patch(facecolor=colors['modeling'], label='Modeling'),
        mpatches.Patch(facecolor=colors['output'], label='Output')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    ax.set_title('Istanbul Rent Prediction Project Pipeline', fontsize=16, weight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(save_path, dpi=config.VISUALIZATION['dpi'], bbox_inches='tight')
    plt.close()
    print(f"Saved project pipeline diagram to: {save_path}")


def generate_pipeline_diagrams():
    """Generate all pipeline diagrams"""
    print("\n=== Generating Pipeline Diagrams ===")
    create_methodology_diagram(config.FIGURES_DIR / "00_methodology_diagram.png")
    create_project_pipeline_diagram(config.FIGURES_DIR / "00_project_pipeline.png")
    print("=== Pipeline Diagrams Complete ===\n")

