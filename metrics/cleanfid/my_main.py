from cleanfid import fid
import torch
import multiprocessing


print(torch.cuda.is_available())
fdir_ref='C:\PhD\SR\SR_averaged_paper_results\HR_of_averaged_equal_hist'
fdir_source='C:\PhD\SR\SR_averaged_paper_results\WIENER_5_5'

# custom_name1='C:\PhD\SR\CYCLEGAN/results/statistics_testHR.npz'
# custom_name2='C:\PhD\SR\CYCLEGAN/results/statistics_testLR.npz'
# custom_name3='C:\PhD\SR\CYCLEGAN/results/statistics_testSR_452x308.npz'

#fid.make_custom_stats(custom_name3, fdir3, mode="clean")



def run():
    #torch.multiprocessing.freeze_support()
    #print('loop')
    score = fid.compute_fid(fdir_source, fdir_ref, mode="clean",num_workers=0)
    print("Score between HR and LR is: ",score)

if __name__ == '__main__':
    run()


