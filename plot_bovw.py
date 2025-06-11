import matplotlib.pyplot as plt

ks = [50, 100, 300, 500, 700, 1000, 1500, 2000]

# BoVW baseline (best & worst)
bovw_best_err = [0.12, 0.11]
bovw_best_time = [0.5744, 0.4398]
bovw_worst_err = [0.36]
bovw_worst_time = [0.5525]

fig, ax1 = plt.subplots(figsize=(8, 5))

# Distance error (left y‑axis)
ax1.set_xlabel('clustering number')
ax1.set_ylabel('Distance error')
ax1.scatter(ks, bovw_best_err, marker='D', color='blue', label='BoVW‑best Distance error (m)')
ax1.scatter(ks, bovw_worst_err, marker='v',  color='red', label='BoVW‑worst Distance error (m)')
ax1.set_xticks(ks)

# Execution time (right y‑axis)
ax2 = ax1.twinx()
ax2.set_ylabel('Execution time')

ax2.scatter(ks, bovw_best_time, marker='D', color='skyblue', label='BoVW‑best Execution time (sec)')
ax2.scatter(ks, bovw_worst_time, marker='v', color='pink', label='BoVW‑worst Execution time (sec)')
ax2.tick_params(axis='y')

# Combine legends from both axes
handles1, labels1 = ax1.get_legend_handles_labels()
handles2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(handles1 + handles2, labels1 + labels2, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)

plt.grid(True)
plt.tight_layout()
plt.savefig('./plot.png', dpi=300, bbox_inches='tight')