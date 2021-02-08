import matplotlib.pyplot as plt

with open('/home/bmil10/Downloads/1/Accel_PPO.csv') as f:
    ppolines = f.read().split('\n')

with open('/home/bmil10/Downloads/1/Accel_DDPG.csv') as f:
    ddpglines = f.read().split('\n')

with open('/home/bmil10/Downloads/1/Accel_TD3.csv') as f:
    td3lines = f.read().split('\n')

ppoaccels = [float(a) for a in ppolines[1:-1]]
ddpgaccels = [float(a) for a in ddpglines[1:-1]]
td3accels = [float(a) for a in td3lines[1:-1]]

plt.plot(ppoaccels)
plt.plot(ddpgaccels)
plt.plot(td3accels)

plt.show()