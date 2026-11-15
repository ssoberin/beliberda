"""Render saved metrics only; no fitting or test rescoring."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path('output/hard-negatives-v3-experiment');s=json.loads((P/'metrics.json').read_text())
fig,axes=plt.subplots(1,2,figsize=(11,4))
keys=['pub_validation','v3_test','official_208']
for shift,name,color in [(-.18,'A','#64748b'),(.18,'B','#0f766e')]:
 bars=axes[0].bar([i+shift for i in range(3)],[s[k][name]['micro_f1'] for k in keys],width=.35,label=name,color=color);axes[0].bar_label(bars,fmt='%.4f',fontsize=8,padding=3)
 axes[1].bar([i+shift for i in range(3)],[s[k][name]['fp'] for k in keys],width=.35,label=name,color=color)
for ax in axes:ax.set_xticks(range(3),['Public val729','V3 test600','Official208*']);ax.spines[['top','right']].set_visible(False);ax.legend()
axes[0].set_ylim(0,1.05);axes[0].set_ylabel('Micro-F1');axes[1].set_ylabel('False positives')
fig.tight_layout();fig.savefig(P/'comparison.png',dpi=150)
