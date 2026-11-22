import json, numpy as np, matplotlib, sys
matplotlib.use("Agg"); import matplotlib.pyplot as plt
R=json.load(open(sys.argv[1] if len(sys.argv)>1 else "macro_run.json"))
t=np.array([r['t'] for r in R])
k=lambda n: np.array([r[n] for r in R])
fig,ax=plt.subplots(2,3,figsize=(15,6.5))
ax[0,0].plot(t,100*k('u')); ax[0,0].set_title('unemployment %')
ax[0,1].plot(t,400*k('pi'),lw=.6,label='pi (ann.)'); ax[0,1].plot(t,400*k('i'),label='policy rate (ann.)'); ax[0,1].legend(); ax[0,1].set_title('rates')
ax[0,2].plot(t,k('I')/k('Y')); ax[0,2].set_title('I/Y')
ax[1,0].plot(t,k('inv')/k('Y')); ax[1,0].set_title('inventory / sales')
ax[1,1].plot(t,k('debt')/(4*k('GDP')),label='gov debt/GDP'); ax[1,1].plot(t,k('A')/(4*k('GDP')),label='retirement/GDP'); ax[1,1].plot(t,k('loans')/(4*k('GDP')),label='loans/GDP');ax[1,1].legend()
ax[1,2].plot(t,k('reserves')); ax[1,2].set_title('bank reserves (should be 0)')
fig.tight_layout(); fig.savefig(sys.argv[2] if len(sys.argv)>2 else "fig_macro_run.png",dpi=90)
