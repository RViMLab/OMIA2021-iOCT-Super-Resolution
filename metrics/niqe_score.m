clear all

% Define the paths of two Datasets.
% One of them (reference) will be the datasets that will train the NIQE model.

path_to_source="/source/"
path_to_reference="/reference/"


%clear all
% Define on which dataset the model will be trained 
train_folder=path_to_reference
imds_train = imageDatastore(train_folder,'FileExtensions',{'.png'});
% Uncomment the following line if you desire to train a new model
model = fitniqe(imds_train);


process_path=path_to_source;
LR_NIQE=getNIQE(process_path,model);


function mean_niqe = getNIQE(process_path,model)

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





