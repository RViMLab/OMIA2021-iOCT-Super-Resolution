import pandas as pd
import matplotlib.pyplot as plt
path_txt = '../pytorch-CycleGAN-and-pix2pix/results/oct_cyclegan/evaluation_results.txt'

file1 = open(path_txt, 'r')
lines = file1.readlines()
dicts = list()
for i, line in enumerate(lines):
    parts = line.split(' ')
    print(parts)

    dict_tmp = dict()
    dict_tmp['Epoch'] = float(parts[1])
    dict_tmp['SR_PSNR'] = float(parts[3])
    dict_tmp['SR_SSIM'] = float(parts[5])
    dict_tmp['SR_lfeat'] = float(parts[7])
    dict_tmp['SR_FID'] = float(parts[9])
    dict_tmp['SR_GCF'] = float(parts[11])

    dict_tmp['LR_PSNR'] = float(parts[13])
    dict_tmp['LR_SSIM'] = float(parts[15])
    dict_tmp['LR_lfeat'] = float(parts[17])
    dict_tmp['LR_FID'] = float(parts[19])
    dict_tmp['LR_GCF'] = float(parts[21])

    dicts.append(dict_tmp)

df1 = pd.DataFrame(dicts)
print(df1)

path_matlab_niqe_txt = '../pytorch-CycleGAN-and-pix2pix/results/oct_cyclegan/niqe_eval_results.txt'
file1 = open(path_matlab_niqe_txt, 'r')
lines = file1.readlines()
dicts2 = list()

for i, line in enumerate(lines):
    parts = line.split(' ')
    dict_tmp = dict()
    #dict_tmp['Epoch'] = float(parts[1])
    dict_tmp['SR_NIQE'] = float(parts[3])
    dict_tmp['LR_NIQE'] = float(parts[5])
    dicts2.append(dict_tmp)

df2 = pd.DataFrame(dicts2)
print(df2)
frames = [df1, df2]
df = pd.concat(frames,axis=1)
print(df)
#plt.figure();
#df['D_A'].plot( label='Series'); plt.legend()
#plt.show()
#plt.figure();
#df.plot(subplots=True, figsize=(6, 6)); plt.legend(loc='best')
#plt.show()

fig, axes = plt.subplots(nrows=3, ncols=2)
axes[0,0].plot(df['Epoch'],df['SR_PSNR'],label='SR'); axes[0,0].set_title('SR-LR_PSNR')
axes[0,0].plot(df['Epoch'],df['LR_PSNR'],color='orange',label='LR');
axes[0,0].legend()
axes[0,1].plot(df['Epoch'],df['SR_SSIM'],label='SR'); axes[0,1].set_title('SR-LR_SSIM')
axes[0,1].plot(df['Epoch'],df['LR_SSIM'],color='orange',label='LR');
axes[0,1].legend()
axes[1,0].plot(df['Epoch'],df['SR_lfeat'],label='SR'); axes[1,0].set_title('SR-LR_lfeat')
axes[1,0].plot(df['Epoch'],df['LR_lfeat'],color='orange',label='LR');
axes[1,0].legend()
axes[1,1].plot(df['Epoch'],df['SR_FID'],label='SR'); axes[1,1].set_title('SR-LR_FID')
axes[1,1].plot(df['Epoch'],df['LR_FID'],color='orange',label='LR');
axes[1,1].legend()

axes[2,0].plot(df['Epoch'],df['SR_GCF'],label='SR'); axes[2,0].set_title('SR-LR_GCF')
axes[2,0].plot(df['Epoch'],df['LR_GCF'],color='orange',label='LR');
axes[2,0].legend()
axes[2,1].plot(df['Epoch'],df['SR_NIQE'],label='SR'); axes[2,1].set_title('SR-LR_NIQE')
axes[2,1].plot(df['Epoch'],df['LR_NIQE'],color='orange',label='LR');
axes[2,1].legend()

plt.show()