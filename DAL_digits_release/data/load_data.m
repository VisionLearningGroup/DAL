clear
train = load('synth_train_32x32.mat');
train_data = train.X(:,:,:,1:25000);
train_label = train.y(1:25000);
train_data = permute(train_data, [4,1,2,3]);

test = load('synth_test_32x32.mat');
test_data = test.X(:,:,:,1:9000);
test_label = test.y(1:9000);
test_data = permute(test_data, [4,1,2,3]);

size(train_data)
size(test_data)


a = load('mnistm_with_label.mat');
save('syn_number.mat', 'train_data','train_label','test_data','test_label');
size(a.train)