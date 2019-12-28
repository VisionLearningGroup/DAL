% svhn_32_ld = load('./Digit-Five/svhn_test_32x32.mat'); */
svhn_32_ld = load('./Digit-Five/svhn_train_32x32.mat');
X = svhn_32_ld.X;
y = svhn_32_ld.y;
[w,h,d,n] = size(X);
len = length(y);
X_final = zeros(28,28,d,len);

for i=1:len
    im = X(:,:,:,i);

    %print(size(im));
    %fprintf('%d %d %d\n', size(im, 1), size(im,2),size(im,3))
    im = imresize(im, [28,28], 'cubic');
    X_final(:,:,:,i) = im;

end

X_final = uint8(X_final);
X = X_final;
y = y(1:len);
% save('-v6', './Digit-Five/svhn_test_28x28.mat', 'X', 'y');
save( '-v6', './Digit-Five/svhn_train_28x28.mat', 'X', 'y');
