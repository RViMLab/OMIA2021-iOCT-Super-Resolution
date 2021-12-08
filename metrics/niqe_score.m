% This is a reference-free metric to assess the quality of the OCT images
% It has been used for natural images but histogram of MSCN (see paper)
% seem to follow a Gaussian. 


clear all
restults_dir="C:\PhD\SR\CYCLEGAN_PIX2PIX_TORCH\pytorch-CycleGAN-and-pix2pix/results/";
name_dir="oct_cyclegan/";
phase_dir="val";


latest = false ; % todo define if we want only the latest saved model or all
min_epochs = 5;  % todo define the minimum checkpoints we have
max_epochs = 30; % todo define the maximum checkpoints we have
epoch=[];
if (latest)
    epoch=[epoch "latest"];

else
    num_of_saved_chkpoints = (max_epochs-min_epochs) / 5;
    for i=min_epochs:5:max_epochs
        epoch(end+1)=i;
    end
end

path_to_normal_real_A=restults_dir+name_dir+phase_dir+"_"+int2str(epoch(1))+"/norm_real_A/"
path_to_normal_real_B=restults_dir+name_dir+phase_dir+"_"+int2str(epoch(1))+"/norm_real_B/"
path_to_normal_fake_B=restults_dir+name_dir+phase_dir+"_"+int2str(epoch(1))+"/norm_fake_B/"

%clear all
% Define on which dataset the model will be trained 
train_folder=path_to_normal_real_B
imds_train = imageDatastore(train_folder,'FileExtensions',{'.png'});
% Uncomment the following line if you desire to train a new model
model = fitniqe(imds_train);


output_dir = restults_dir + name_dir;

fileID = fopen(output_dir+'niqe_eval_results.txt','w');

for k=1:size(epoch(:))
    
path_to_normal_real_A=restults_dir+name_dir+phase_dir+"_"+int2str(epoch(k))+"/norm_real_A/"
path_to_normal_fake_B=restults_dir+name_dir+phase_dir+"_"+int2str(epoch(k))+"/norm_fake_B/"


process_path=path_to_normal_real_A;
LR_NIQE=getNIQE(process_path,model);

process_path=path_to_normal_fake_B;
SR_NIQE=getNIQE(process_path,model);

fprintf(fileID,'Epoch: %s SR_NIQE: %f LR_NIQE: %f\n',int2str(epoch(k)), SR_NIQE,  LR_NIQE);


end

fclose(fileID);

function mean_niqe = getNIQE(process_path,model)
    % Read images and initialize
imds_process = imageDatastore(process_path,'FileExtensions',{'.png'});
imgs_process= readall(imds_process);
count=0;
score_niqe=0;
score_niqe_matrix = zeros(193,1);
for k = 1 : length(imgs_process)
    score_niqe=niqe(imgs_process{k},model);
    score_niqe_matrix(k,1)=score_niqe;

    count=count+score_niqe;
end

niqe_averaged=count/length(imgs_process);
mean_niqe=mean(score_niqe_matrix);
std_niqe=std(score_niqe_matrix);
end




