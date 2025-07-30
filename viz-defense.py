import matplotlib.pyplot as plt
import numpy as np

# Data
method = 'taba'
# method = 'flair' 
# method = 'flair+'

stages = np.arange(1, 11)  # 0-9 stages

if method == 'taba':
    fgsm_asr_raw = [27.1739, 25.9398, 25.2500, 30.4167, 30.2920,
                33.6066, 33.9678, 40.0541, 42.5000, 46.3547]
    pgd_asr_raw = [22.8261, 19.9248, 21.2500, 23.9583, 24.4526,
               26.0656, 27.5256, 32.6116, 33.6842, 37.8672]
    auto_asr_raw = [7.6087, 4.8872, 5.5000, 5.8333, 6.9343,
                8.1967, 9.5168, 17.5913, 21.4474, 65.6148]
    fgsm_asr = [x/100 for x in fgsm_asr_raw]
    pgd_asr = [x/100 for x in pgd_asr_raw]
    auto_asr = [x/100 for x in auto_asr_raw]
elif method == 'flair':
    fgsm_asr_raw = [28.2667, 37.1166, 35.6637, 41.2335, 40.8822,
                43.0964, 44.6882, 46.7369, 48.8063, 52.4734]
    pgd_asr_raw = [24.2667, 30.5215, 29.8230, 33.8184, 33.6202,
               34.8574, 36.0662, 37.9674, 40.0725, 43.7500]
    auto_asr_raw = [4.6729, 6.7797, 7.3663, 7.7640, 8.3724,
                10.6609, 12.1529, 15.0141, 18.1650, 62.3447]
    fgsm_asr = [x/100 for x in fgsm_asr_raw]
    pgd_asr = [x/100 for x in pgd_asr_raw]
    auto_asr = [x/100 for x in auto_asr_raw]
elif method == 'flair+':
    fgsm_asr_raw = [24.9221, 33.7100, 30.1715, 35.0155, 36.3583,
                38.4467, 39.2757, 41.7018, 43.5897, 47.8120]
    pgd_asr_raw = [19.0031, 26.9303, 24.3189, 27.8727, 29.5082,
               30.9214, 31.9517, 34.2827, 36.2373, 39.5462]
    auto_asr_raw = [4.7829, 6.6597, 7.3854, 7.5663, 8.2967,
                10.7809, 11.9122, 15.2044, 18.0141, 62.4377]
    fgsm_asr = [x/100 for x in fgsm_asr_raw]
    pgd_asr = [x/100 for x in pgd_asr_raw]
    auto_asr = [x/100 for x in auto_asr_raw]

plt.figure(figsize=(5, 4))

# Plot lines
plt.plot(stages, fgsm_asr, 'o-', label='FGSM', color='tab:blue', linewidth=2)
plt.plot(stages, pgd_asr, 'o-', label='PGD', color='tab:orange', linewidth=2)
plt.plot(stages, auto_asr, 'o-', label='AutoAttack', color='tab:green', linewidth=2)

# Customize plot
plt.xlabel('Stage', fontsize=12)
plt.ylabel('ASR', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=10)
plt.title(f'{method} Defense', fontsize=12)

# Set x-ticks
plt.xticks(stages)

# Adjust layout and save
plt.tight_layout()
plt.savefig('./results/defense_asr.png', dpi=300, bbox_inches='tight')
plt.close()
print("Plot saved as defense_asr.png")
