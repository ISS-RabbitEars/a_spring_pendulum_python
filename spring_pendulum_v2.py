import numpy as np
import sympy as sp
from sympy.physics.vector import dynamicsymbols
from scipy.integrate import odeint
from matplotlib import pyplot as plt
from matplotlib import animation


def integrate(ic, ti, p):
	gc, m, k, req, xp, yp = p
	r, v, theta, omega = ic

	sub = {g:gc, M:m, K:k, Req:req, Xp:xp, Yp:yp, R:r, Rdot:v, THETA:theta, THETAdot:omega}

	diff_eq = [v,A.subs(sub),omega,ALPHA.subs(sub)]

	print(ti)

	return diff_eq


M, K, Req, Xp, Yp, g, t = sp.symbols('M K Req Xp Yp g t')
R, THETA = dynamicsymbols('R THETA')

Rdot = R.diff(t, 1)
Rddot = R.diff(t, 2)
THETAdot = THETA.diff(t, 1)
THETAddot = THETA.diff(t, 2)

X = Xp + R * sp.cos(THETA)
Y = Yp + R * sp.sin(THETA)
dR = sp.sqrt((X - Xp)**2 + (Y - Yp)**2)

Xdot = X.diff(t, 1)
Ydot = Y.diff(t, 1)

T = sp.simplify(sp.Rational(1, 2) * M * (Xdot**2 + Ydot**2))
V = sp.simplify(sp.Rational(1, 2) * K * (dR - Req)**2 + M * g * Y)

L = T - V

dLdR = L.diff(R, 1)
dLdRdot = L.diff(Rdot, 1)
ddtdLdRdot = dLdRdot.diff(t, 1)
dL = ddtdLdRdot - dLdR

dLdTHETA = L.diff(THETA, 1)
dLdTHETAdot = L.diff(THETAdot, 1)
ddtdLdTHETAdot = dLdTHETAdot.diff(t, 1)
dTHETA = ddtdLdTHETAdot - dLdTHETA

sol = sp.solve([dL,dTHETA], (Rddot,THETAddot))

A = sp.simplify(sol[Rddot])
ALPHA = sp.simplify(sol[THETAddot])

#------------------------------------------------- 

gc = 9.8
m = 1
k = 25
req = 1.5
ro = 3 
vo = 0
thetao = 45
omegao = 0
xp,yp = [1, 1] 
tf = 60 

cnvrt = np.pi/180
thetao *= cnvrt
omegao *= cnvrt

p = gc, m, k, req, xp, yp
ic = ro, vo, thetao, omegao

nfps = 30
nframes = tf * nfps
ta = np.linspace(0, tf, nframes)

rth = odeint(integrate, ic, ta, args = (p,))

x = np.asarray([X.subs({Xp:xp, R:rth[i,0], THETA:rth[i,2]}) for i in range(nframes)],dtype=float)
y = np.asarray([Y.subs({Yp:yp, R:rth[i,0], THETA:rth[i,2]}) for i in range(nframes)],dtype=float)

ke = np.asarray([T.subs({M:m, Xp:xp, Yp:yp, R:rth[i,0], Rdot:rth[i,1], THETA:rth[i,2], THETAdot:rth[i,3]}) for i in range(nframes)],dtype=float)
pe = np.asarray([V.subs({g:gc, M:m, K:k, Req:req, Xp:xp, Yp:yp, R:rth[i,0], Rdot:rth[i,1], THETA:rth[i,2], THETAdot:rth[i,3]}) for i in range(nframes)],dtype=float)
E = ke + pe

#---------------------------------------------------

xmax = x.max() if x.max() > xp else xp
xmin = x.min() if x.min() < xp else xp
ymax = y.max() if y.max() > yp else yp
ymin = y.min() if y.min() < yp else yp

mrf = 1 / 30
drs = np.sqrt((xmax - xmin)**2 + (ymax - ymin)**2)
mr = mrf * drs

xmax += 2 * mr
xmin -= 2 * mr
ymax += 2 * mr
ymin -= 2 * mr

dr = np.asarray([np.sqrt((xp - x[i])**2 + (yp - y[i])**2) for i in range(nframes)],dtype=float)
drmax = dr.max()
theta = np.asarray([np.arccos((yp - y[i])/dr[i]) for i in range(nframes)],dtype=float)
nl = int(np.ceil(drmax/(2 * mr)))
l = np.asarray([(dr[i] - mr)/nl for i in range(nframes)],dtype=float)
h = np.asarray([np.sqrt(mr**2 - (0.5 * l[i])**2) for i in range(nframes)],dtype=float)
flipa = np.zeros(nframes)
flipb = np.zeros(nframes)
flipc = np.zeros(nframes)
flipa = np.asarray([-1 if x[j]>xp and y[j]<yp else 1 for j in range(nframes)])	
flipb = np.asarray([-1 if x[j]<xp and y[j]>yp else 1 for j in range(nframes)])
flipc = np.asarray([-1 if x[j]<xp else 1 for j in range(nframes)])
xlo = np.asarray([x[i] + np.sign((yp - y[i]) * flipa[i] * flipb[i]) * mr * np.sin(theta[i]) for i in range(nframes)])
ylo = np.asarray([y[i] + mr * np.cos(theta[i]) for i in range(nframes)])
xl = np.zeros((nl,nframes))
yl = np.zeros((nl,nframes))
for i in range(nframes):
	for j in range(nl):
		xl[j][i] = xlo[i] + np.sign((yp - y[i])*flipa[i]*flipb[i]) * (0.5 + j) * l[i] * np.sin(theta[i]) - np.sign((yp - y[i])*flipa[i]*flipb[i]) * flipc[i] * (-1)**j * h[i] * np.sin(np.pi/2 - theta[i])
		yl[j][i] = ylo[i] + (0.5 + j) * l[i] * np.cos(theta[i]) + flipc[i] * (-1)**j * h[i] * np.cos(np.pi/2 - theta[i])

fig, a=plt.subplots()

def run(frame):
	plt.clf()
	plt.subplot(211)
	circle=plt.Circle((x[frame],y[frame]),radius=mr,fc='xkcd:red')
	plt.gca().add_patch(circle)
	circle=plt.Circle((xp,yp),radius=mr/2,fc='xkcd:cerulean')
	plt.gca().add_patch(circle)
	plt.plot([xlo[frame],xl[0][frame]],[ylo[frame],yl[0][frame]],'xkcd:cerulean')
	plt.plot([xl[nl-1][frame],xp],[yl[nl-1][frame],yp],'xkcd:cerulean')
	for i in range(nl-1):
		plt.plot([xl[i][frame],xl[i+1][frame]],[yl[i][frame],yl[i+1][frame]],'xkcd:cerulean')
	plt.title("Spring Pendulum")
	ax=plt.gca()
	ax.set_aspect(1)
	plt.xlim([float(xmin),float(xmax)])
	plt.ylim([float(ymin),float(ymax)])
	ax.xaxis.set_ticklabels([])
	ax.yaxis.set_ticklabels([])
	ax.xaxis.set_ticks_position('none')
	ax.yaxis.set_ticks_position('none')
	ax.set_facecolor('xkcd:black')
	plt.subplot(212)
	plt.plot(ta[0:frame],ke[0:frame],'xkcd:red',lw=1.0)
	plt.plot(ta[0:frame],pe[0:frame],'xkcd:cerulean',lw=1.0)
	plt.plot(ta[0:frame],E[0:frame],'xkcd:bright green',lw=1.5)
	plt.xlim([0,tf])
	plt.title("Energy")
	ax=plt.gca()
	ax.legend(['T','V','E'],labelcolor='w',frameon=False)
	ax.set_facecolor('xkcd:black')

ani=animation.FuncAnimation(fig,run,frames=nframes)
writervideo = animation.FFMpegWriter(fps=nfps)
ani.save('spring_pendulum_v2.mp4', writer=writervideo)
plt.show()











