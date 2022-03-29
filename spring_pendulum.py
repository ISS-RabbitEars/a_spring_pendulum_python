import numpy as np
import sympy as sp
from sympy.physics.vector import dynamicsymbols
from scipy.integrate import odeint
from matplotlib import pyplot as plt
from matplotlib import animation

def integrate(ic, ti, p):
	m, k, req, gc = p
	r, v, theta, omega = ic

	print(ti)

	return[v,A.subs({M:m, K:k, R:r, REQ:req, THETA:theta, THETAdot:omega, g:gc}),\
		omega,ALPHA.subs({R:r, Rdot:v, THETA:theta, THETAdot:omega, g:gc})]


M, K, REQ, g, t = sp.symbols('M K REQ g t')
R, THETA = dynamicsymbols('R THETA')

X = R * sp.sin(THETA)
Y = - R * sp.cos(THETA)

Xdot = X.diff(t, 1)
Ydot = Y.diff(t, 1)

Vs = Xdot**2 + Ydot**2

T = sp.Rational(1, 2) * M * Vs
V = sp.Rational(1, 2) * K * (R - REQ)**2 + M * g * Y

L = T - V

Rdot = R.diff(t, 1)
dLdR = L.diff(R, 1)
dLdRdot = L.diff(Rdot, 1)
ddtdLdRdot = dLdRdot.diff(t, 1)

dLR = ddtdLdRdot - dLdR

THETAdot = THETA.diff(t, 1)
dLdTHETA = L.diff(THETA, 1)
dLdTHETAdot = L.diff(THETAdot, 1)
ddtdLdTHETAdot = dLdTHETAdot.diff(t, 1)

dLTHETA = ddtdLdTHETAdot - dLdTHETA

sol_R = sp.solve(dLR, R.diff(t, 2))
sol_THETA = sp.solve(dLTHETA, THETA.diff(t, 2))

A = sp.simplify(sol_R[0])
ALPHA = sp.simplify(sol_THETA[0])

#-----------------------------------------------

gc = 9.8
m = 1
k = 50
req = 1
ro = 1
vo = 0
thetao = 135
omegao = 0
thetao *= np.pi/180
omegao *= np.pi/180


tf = 60
nfps = 30
nframes = tf * nfps
ta = np.linspace(0, tf, nframes)


p = m, k, req, gc
ic = ro, vo, thetao, omegao


rth = odeint(integrate, ic, ta, args=(p,))


x = np.asarray([X.subs({R:rth[i,0], THETA:rth[i,2]}) for i in range(nframes)])
y = np.asarray([Y.subs({R:rth[i,0], THETA:rth[i,2]}) for i in range(nframes)])

ke = np.asarray([T.subs({M:m, R:rth[i,0], Rdot:rth[i,1], THETA:rth[i,2], THETAdot:rth[i,3]}) for i in range(nframes)])
pe = np.asarray([V.subs({M:m, K:k, REQ:req, g:gc, R:rth[i,0], THETA:rth[i,2]}) for i in range(nframes)])
E = ke + pe

#---------------------------------------------------

rad = 0.25
xmax = float(x.max() + 2 * rad)
xmin = float(x.min() - 2 * rad)
ymax = float(y.max() + 2 * rad)
if ymax < 0:
	ymax = 2 * rad
ymin = float(y.min() - 2 * rad)
rmax = max(np.abs(rth[:,0]))
nl = int(np.ceil(rmax/(2*rad)))
l = (np.asarray(np.abs(rth[:,0]))-rad)/nl
h = np.sqrt(rad**2 - (0.5*l)**2)
xlo = x - rad*np.cos(np.pi/2 - np.asarray(rth[:,2]))
ylo = y + rad*np.sin(np.pi/2 - np.asarray(rth[:,2]))
xl = np.zeros((nl,nframes))
yl = np.zeros((nl,nframes))
for i in range(nframes):
	xl[0][i] = xlo[i] - 0.5 * l[i] * np.cos(np.pi/2 - rth[i,2]) + h[i] * np.cos(rth[i,2])
	yl[0][i] = ylo[i] + 0.5 * l[i] * np.sin(np.pi/2 - rth[i,2]) + h[i] * np.sin(rth[i,2])
for j in range(nframes):
	for i in range(1,nl):
		xl[i][j] = xlo[j] - (0.5 + i) * l[j] * np.cos(np.pi/2 - rth[j,2]) + (-1)**i * h[j] * np.cos(rth[j,2])
		yl[i][j] = ylo[j] + (0.5 + i) * l[j] * np.sin(np.pi/2 - rth[j,2]) + (-1)**i * h[j] * np.sin(rth[j,2])
fig, a=plt.subplots()

def run(frame):
	plt.clf()
	plt.subplot(211)
	circle=plt.Circle((x[frame],y[frame]),radius=rad,fc='xkcd:red')
	plt.gca().add_patch(circle)
	plt.plot([xlo[frame],xl[0][frame]],[ylo[frame],yl[0][frame]],'xkcd:cerulean')
	plt.plot([xl[nl-1][frame],0],[yl[nl-1][frame],0],'xkcd:cerulean')
	for i in range(nl-1):
		plt.plot([xl[i][frame],xl[i+1][frame]],[yl[i][frame],yl[i+1][frame]],'xkcd:cerulean')
	plt.title("A Spring Pendulum")
	ax=plt.gca()
	ax.set_aspect(1)
	plt.xlim([xmin,xmax])
	plt.ylim([ymin,ymax])
	ax.xaxis.set_ticklabels([])
	ax.yaxis.set_ticklabels([])
	ax.xaxis.set_ticks_position('none')
	ax.yaxis.set_ticks_position('none')
	ax.set_facecolor('xkcd:black')
	plt.subplot(212)
	plt.plot(ta[0:frame],ke[0:frame],'xkcd:red',lw=0.5)
	plt.plot(ta[0:frame],pe[0:frame],'xkcd:cerulean',lw=0.5)
	plt.plot(ta[0:frame],E[0:frame],'xkcd:bright green',lw=1.0)
	plt.xlim([0,tf])
	plt.title("Energy")
	ax=plt.gca()
	ax.legend(['T','V','E'],labelcolor='w',frameon=False)
	ax.set_facecolor('xkcd:black')

ani=animation.FuncAnimation(fig,run,frames=nframes)
writervideo = animation.FFMpegWriter(fps=nfps)
ani.save('spring_pendulum.mp4', writer=writervideo)
plt.show()



