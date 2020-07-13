
n_v = size(W,1);

degree = sum(W,2);
D = diag(degree);

D_rsqr = 1./sqrt(D);
D_rsqr(D_rsqr==Inf) = 0;

L = D - W;
L_sym = eye(n_v) - D_rsqr * W * D_rsqr;
L_sym = D_rsqr*L*D_rsqr;

%[U,u] = eig(L_sym);
%u = diag(u);
[U,u] = svd(L_sym);
u = diag(u);
[u,idx] = sort(u,'ascend');
U = U(:,idx);

k=10;
T = U(:,1:k);
for r=1:size(U,1)
    nrm = sqrt(sum(T(r,:).^2,2));    
    for c=1:k
        if nrm ~= 0
            T(r,c) = T(r,c) ./ nrm;
        else
            T(r,c) = 0;
        end        
    end
end
IDX = kmeans(T,k);

IDX22 = kmeans(T2,k);
figure(1);
subplot(1,3,1);
colors = hsv(k);
coeff = pca(T);
pts = T*coeff(:,1:3);
hold on;
for i=1:k     
    scatter3(pts(IDX==i,1),pts(IDX==i,2),pts(IDX==i,3),20,colors(i,:));    
end
hold off;

subplot(1,3,2);
colors = hsv(k);
coeff = pca(T2);
pts = T2*coeff(:,1:3);
hold on;
for i=1:k     
    scatter3(pts(IDX2==i-1,1),pts(IDX2==i-1,2),pts(IDX2==i-1,3),20,colors(i,:));    
end
hold off;

subplot(1,3,3);
colors = hsv(k);
coeff = pca(T2);
pts = T2*coeff(:,1:3);
hold on;
for i=1:k     
    scatter3(pts(IDX22==i,1),pts(IDX22==i,2),pts(IDX22==i,3),20,colors(i,:));    
end
hold off;


IDX22 = kmeans(T2,k);
figure(1);
subplot(1,3,1);
colors = hsv(k);
coeff = pca(T);
pts = T*coeff(:,1:3);
hold on;
for i=1:k     
    scatter3(pts(IDX==i,1),pts(IDX==i,2),pts(IDX==i,3),20,colors(i,:));    
end
hold off;

subplot(1,3,2);
colors = hsv(k);
coeff = pca(T2);
pts = T2*coeff(:,1:3);
hold on;
for i=1:k     
    scatter3(pts(IDX2==i-1,1),pts(IDX2==i-1,2),pts(IDX2==i-1,3),20,colors(i,:));    
end
hold off;

subplot(1,3,3);
colors = hsv(k);
coeff = pca(T2);
pts = T2*coeff(:,1:3);
hold on;
for i=1:k     
    scatter3(pts(IDX22==i,1),pts(IDX22==i,2),pts(IDX22==i,3),20,colors(i,:));    
end
hold off;
