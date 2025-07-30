import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--method', type=str, default='icarl')
args = parser.parse_args()

# Data for all methods
data = {
    'iCaRL': {
        'cos_sim': [0.9852, 0.9896, 0.9932, 0.9960],
        'cka_sim': [0.9744, 0.9814, 0.9833, 0.9851],
        'asr': [x/0.953 for x in [0.774, 0.895, 0.927, 0.941]]
    },
    'GDumb': {
        'cos_sim': [0.8781, 0.8827, 0.8972, 0.9186],
        'cka_sim': [0.9675, 0.9790, 0.9794, 0.9934],
        'asr': [x/0.127 for x in [0.072, 0.078, 0.114, 0.110]]
    },
    'ER-ACE': {
        'cos_sim': [0.6607, 0.7839, 0.8733, 0.9370],
        'cka_sim': [0.9608, 0.9678, 0.9691, 0.9831],
        'asr': [x/0.155 for x in [0.087, 0.085, 0.122, 0.126]]
    },
    'ER-AML': {
        'cos_sim': [0.3619, 0.5970, 0.7732, 0.8593],
        'cka_sim': [0.8945, 0.9173, 0.9620, 0.9503],
        'asr': [x/0.141 for x in [0.020, 0.038, 0.093, 0.129]]
    }
}

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
markers = ['o', 'o', 'o', 'o']

# Plot cosine similarity
for i, (method, color, marker) in enumerate(zip(data.keys(), colors, markers)):
    ax1.plot(data[method]['cos_sim'], data[method]['asr'], 
             color=color, marker=marker, label=method)
    for j, (x, y) in enumerate(zip(data[method]['cos_sim'], data[method]['asr']), 1):
        ax1.annotate(str(j), (x, y), xytext=(5, 5), textcoords='offset points')

ax1.set_xlabel('Similarity')
ax1.set_ylabel('ASR Ratio')
ax1.grid(True, linestyle='--', alpha=0.7)
ax1.set_title('$r=0.92$')
ax1.legend()

# Plot CKA similarity
for i, (method, color, marker) in enumerate(zip(data.keys(), colors, markers)):
    ax2.plot(data[method]['cka_sim'], data[method]['asr'], 
             color=color, marker=marker, label=method)
    for j, (x, y) in enumerate(zip(data[method]['cka_sim'], data[method]['asr']), 1):
        ax2.annotate(str(j), (x, y), xytext=(5, 5), textcoords='offset points')

ax2.set_xlabel('Similarity')
ax2.set_ylabel('ASR Ratio')
ax2.grid(True, linestyle='--', alpha=0.7)
ax2.set_title('$r=0.86$')
ax2.legend()

plt.tight_layout()
plt.savefig('./results/similarity_all.png')
print("Saved at ./results/similarity_all.png")
plt.close()
