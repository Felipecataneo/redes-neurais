# visualizacoes.py

import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from matplotlib.patches import Circle

def plot_perceptron(weights, bias, X, y, title="Perceptron"):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    ax1.set_xlim(0, 10); ax1.set_ylim(0, 6); ax1.set_title("Arquitetura do Perceptron", fontsize=14, fontweight='bold')
    for i in range(2):
        ax1.add_patch(Circle((1, 2 + i * 2), 0.3, color='lightblue', ec='black')); ax1.text(1, 2 + i * 2, f'x{i+1}', ha='center', va='center', fontweight='bold')
        ax1.arrow(1.3, 2 + i * 2, 3.4, 1 - i * 2, head_width=0.1, head_length=0.1, fc='red', ec='red'); ax1.text(3, 2.5 + i * 0.5, f'w{i+1}={weights[i]:.2f}', fontsize=10, color='red')
    ax1.add_patch(Circle((5.5, 3), 0.5, color='orange', ec='black')); ax1.text(5.5, 3, 'Σ', ha='center', va='center', fontsize=16, fontweight='bold')
    ax1.text(5.5, 2.2, f'bias={bias:.2f}', ha='center', fontsize=10, color='blue'); ax1.arrow(6, 3, 2, 0, head_width=0.1, head_length=0.1, fc='green', ec='green')
    ax1.add_patch(Circle((8.5, 3), 0.3, color='lightgreen', ec='black')); ax1.text(8.5, 3, 'y', ha='center', va='center', fontweight='bold')
    ax1.set_aspect('equal'); ax1.axis('off')
    ax2.scatter(X[y == 0, 0], X[y == 0, 1], c='red', marker='o', label='Classe 0', s=50); ax2.scatter(X[y == 1, 0], X[y == 1, 1], c='blue', marker='s', label='Classe 1', s=50)
    if weights[1] != 0:
        x_line = np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 100); y_line = -(weights[0] * x_line + bias) / weights[1]
        ax2.plot(x_line, y_line, 'k--', linewidth=2, label='Linha de Decisão')
    ax2.set_xlabel('Feature 1'); ax2.set_ylabel('Feature 2'); ax2.set_title(title); ax2.legend(); ax2.grid(True, alpha=0.3)
    return fig

def plot_neural_network(layers):
    fig, ax = plt.subplots(figsize=(12, 8)); layer_positions = np.linspace(1, 10, len(layers)); max_neurons = max(layers) if layers else 1; colors = ['lightblue', 'orange', 'lightgreen']
    for i, (pos, neurons) in enumerate(zip(layer_positions, layers)):
        y_positions = np.linspace(1, max_neurons, neurons); color = colors[i % len(colors)]
        for j, y_pos in enumerate(y_positions):
            ax.add_patch(Circle((pos, y_pos), 0.2, color=color, ec='black'))
            if i < len(layers) - 1:
                next_y_positions = np.linspace(1, max_neurons, layers[i + 1])
                for next_y in next_y_positions:
                    ax.plot([pos + 0.2, layer_positions[i + 1] - 0.2], [y_pos, next_y], 'k-', alpha=0.3, linewidth=0.5)
    labels = ['Entrada', 'Oculta', 'Saída'];
    for i, pos in enumerate(layer_positions):
        if i < len(labels): ax.text(pos, max_neurons + 0.5, labels[i], ha='center', fontweight='bold')
    ax.set_xlim(0, 11); ax.set_ylim(0, max_neurons + 1); ax.set_aspect('equal'); ax.axis('off'); ax.set_title('Arquitetura da Rede Neural', fontsize=16, fontweight='bold')
    return fig

def plot_decision_boundary(model, X, y):
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5; y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    fig = go.Figure()
    fig.add_trace(go.Contour(x=xx[0], y=yy[:, 0], z=Z, colorscale='RdBu', showscale=False, opacity=0.8))
    fig.add_trace(go.Scatter(x=X[y==0, 0], y=X[y==0, 1], mode='markers', marker_color='red', name='Classe 0'))
    fig.add_trace(go.Scatter(x=X[y==1, 0], y=X[y==1, 1], mode='markers', marker_color='blue', name='Classe 1'))
    fig.update_layout(title="Fronteira de Decisão da Rede Neural", xaxis_title="Feature 1", yaxis_title="Feature 2")
    return fig