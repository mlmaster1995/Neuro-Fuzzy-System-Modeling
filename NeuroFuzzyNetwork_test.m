%{
~ this code is to take 3000 data pairs and perform prediction test
%}
clc; 
close all; 

%{
~ initialize the traning process params
%}
num_data_pair = 2997;                       %define num of data pair for testing
data_pair_base = 4;                          % define num of data in the pair
input_num = 3;                              % 3 inputs
mf_num_perinput = 2;                        % each input has 2 membership function
mf_num_total = input_num * mf_num_perinput; % total 3*2 =6 mf
rule_num = mf_num_perinput^input_num;       % rule number 2^3=8; 

%{
~ assign the premise and consequent params
%}
num_params_mf = 2; 
mf_params_mat =[   5.1137   -5.0077    7.0944   -7.0263    3.9718   -4.0022
    0.4374   -0.0788    0.4017   -0.0944    0.8527    1.5051];

mf_params_mat_lse = [       5.0000   -5.0000    7.0000   -7.0000    4.0000   -4.0000
    1.0000    1.0000    0.4000    0.4000    1.5000    1.5000];

mf_params_mat_lse_sgd = [        5.0000   -5.0000    7.0000   -7.0000    3.9998   -4.0002
    0.9029    1.0972    0.3833    0.4171    1.4701    1.5287];

[row, mf_num] = size(mf_params_mat);

% ~generate premise data training data pair
p_data_pair = get_premise_training_data(num_data_pair);
[c_data_train,c_data_target,c_data_raw] = get_consequent_training_data(p_data_pair);

new_mat =    [ 0.1460    0.5331    0.0006    0.0021    0.0537    0.1961    0.0002    0.0007
    0.1445    0.5278    0.0006    0.0021    0.0532    0.1942    0.0002    0.0007
    0.1431    0.5226    0.0005    0.0021    0.0527    0.1923    0.0002    0.0007
   -0.0556   -0.2030   -0.0002   -0.0008   -0.0205   -0.0747   -0.0001   -0.0003];

new_mat_lse =[    0.2960    0.3731    0.0017   -0.0059    0.0527    0.3596    0.0000    0.3804
    0.2930    0.3694    0.0017   -0.0058    0.0521    0.3561    0.0000    0.3766
    0.2901    0.3658    0.0017   -0.0058    0.0516    0.3525    0.0000    0.3728
    0.0658   -0.0614    0.0009    0.0044    0.0141   -0.0305    0.0041   -0.0368];

new_mat_lse_sgd =[    0.1192    0.2911   -0.0692   -0.2570    0.1900    0.3290   -0.0632    0.2997
    0.1181    0.2882   -0.0685   -0.2544    0.1881    0.3257   -0.0626    0.2968
    0.1169    0.2853   -0.0678   -0.2519    0.1863    0.3225   -0.0619    0.2938
    1.0395    0.0717   -0.0725    1.0341    0.3424    0.0177    0.0045   -0.0058];
  
    for i=1: num_data_pair
        %************************* layer 1
        O1 = layer1_output(p_data_pair(i,:), mf_num_perinput, input_num, mf_params_mat, num_params_mf);
        O1_ = layer1_output(p_data_pair(i,:), mf_num_perinput, input_num, mf_params_mat_lse, num_params_mf);
        O1__ = layer1_output(p_data_pair(i,:), mf_num_perinput, input_num, mf_params_mat_lse_sgd, num_params_mf);
        
        %************************* layer2
        O2 = layer2_output(O1',mf_num_perinput, rule_num, input_num);
        O2_ = layer2_output(O1_',mf_num_perinput, rule_num, input_num);
        O2__ = layer2_output(O1__',mf_num_perinput, rule_num, input_num);
        
        %************************ layer3
        O3 = layer3_output(O2);
        O3_ = layer3_output(O2_);
        O3__ = layer3_output(O2__);
        
        %************************ layer4
        O4 = layer4_output(O3, c_data_train(i,:), new_mat, rule_num);
        O4_ = layer4_output(O3_, c_data_train(i,:), new_mat_lse, rule_num);   
        O4__ = layer4_output(O3__, c_data_train(i,:), new_mat_lse_sgd, rule_num);
        
        %************************ layer5
        O5 = layer5_output(O4);
        O5_ = layer5_output(O4_); 
        O5__ = layer5_output(O4__);
        
        output(i) = O5;
        output_(i) = O5_;
        output__(i) = O5__;
    end

%{
~ plotting the graphs
%}
x_plot = 1:1:num_data_pair;
error = (output'-c_data_target).^2;
error_ = (output_'-c_data_target).^2;
error__ = (output__'-c_data_target).^2;

subplot(2,1,1)
plot(x_plot,error);
hold on;
plot(x_plot,error_);
hold on;
plot(x_plot,error__,'g');
legend("SGD,no LSE","LSE,no SGD","SGD&LSE");
grid;
title('Error');
xlabel('Data pairs');
ylabel('Error');


subplot(2,1,2);
plot(x_plot, c_data_target);
hold on; 
plot(x_plot, output);
hold on; 
plot(x_plot, output_);
hold on;
plot(x_plot, output__,'g');
legend("Target","SGD,no LSE","LSE,no SGD","LSE&SGD");
grid
title('Prediction');
xlabel('Data pairs');
ylabel('Target');

%---------------------------------Function Definition
%{
% get the updated concequent params
% data_pair format [x,y,z,1]
%}
function [update_mat, update_vec, new_P] = get_consequent_update_params(update_norm, update_P, update_vector, data_pair, data_target, rule_num, input_num)
    % config a&b
    mat_unit = update_norm * data_pair; 
    mat_unit = mat_unit';
    mat_row = mat_unit(:)'; % convert a column to a row
    a_update = mat_row;
    b = data_target;
    
    % update P & params
    I = (update_P * a_update')* a_update * update_P; 
    II = 1 + a_update * update_P * a_update';
    update_P = update_P - I./II;
    update_vector = update_vector + (update_P * a_update' ).* (b - a_update * update_vector);
    
    % convet update_vec to update_mat
    [params_row, params_column] = size(update_vector);
    params_mat_row = input_num+1;
    params_mat_col = params_row/ (input_num+1);
    params_mat = ones(params_mat_row, params_mat_col);
    start = 1; 
    for i=1:params_mat_col
        params_mat(:,i) = update_vector(start:start+3);
        start = start + 4;
    end 
    
    update_mat = params_mat;
    update_vec = update_vector; 
    new_P = update_P; 
end

%*****************************consequent params initilization function
%{
~ get the initial_consequent_params_matrix
~ norm_update -> the norm vector from the layer 3, must be a vector
~ c_data_train -> consequent training data
~ c_data_target -> consequent training data target
~ initial_mat -> initial params matrix
~ initial_vec -> initial params vector
%}
function [initial_mat, initial_vec, update_P] = get_consequent_initial_params(norm_update, c_data_train, c_data_target, rule_num, input_num)
    
    %define initial A & b
    [data_pair_num, data_pair_base] = size(c_data_train);
    A = ones(data_pair_num, rule_num*(input_num+1)); 
    b = c_data_target;
    
    % config A
    for i= 1: data_pair_num
        mat_unit = norm_update*c_data_train(i,:); % outputer product of norm_update vector and data row
        mat_unit = mat_unit';   % transpose row into colum for vecotring all column into one column
        mat_row = mat_unit(:)'; % link all columns into one column & transpose into one row
        A(i,:) = mat_row;
    end

    % calc initial consequent params
    rATA = inv(A'*A-(eye(size(A'*A))*1e-3));
    params_initial = rATA*A'*b;
    [params_row, params_column] = size(params_initial);
    
    % define consequent params matrix row = num of params, column = num of
    % rules
    params_mat_row = input_num+1;
    params_mat_col = params_row/ (input_num+1);
    params_mat = ones(params_mat_row, params_mat_col);
    
    % convert params vecter to a mat
    start = 1; 
    for i=1:params_mat_col
        params_mat(:,i) = params_initial(start:start+3);
        start = start + 4;
    end 
    
    update_P = rATA;
    initial_mat = params_mat;
    initial_vec = params_initial; 
end


%*****************************consequent training data pair generator
%{
% get consequent training data function
% must work with premise training data pair generator function
% c_data_raw -> raw training data set format: [xi,yi,zi,1,ti]
% c_data_train -> training data set format: [x,y,z,1]
% c_data_target -> training data target: format: [t1,t2,t3....]
%}
function [c_data_train,c_data_target,c_data_raw] = get_consequent_training_data(p_data_pair)
    [data_pair_num, data_pair_base] = size(p_data_pair);
    data_mat = ones(data_pair_num, data_pair_base+1);   
    training_mat= ones(data_pair_num,data_pair_base);
    training_target = ones(data_pair_num,1);
    
    % make data matrices
    for i= 1: data_pair_num
        data_mat(i,:) = [p_data_pair(i,1:data_pair_base-1),1,p_data_pair(i,data_pair_base)];
        training_mat(i,:) = [p_data_pair(i,1:data_pair_base-1),1]; 
        training_target(i) = p_data_pair(i,data_pair_base);
    end
      
   c_data_train = training_mat;     % raw data pair
   c_data_target = training_target; % consequent trianing data set format: [x,y,z,1]
   c_data_raw = data_mat; % consequent training target: [t1, t2....]
end

%*****************************premise training data pair generator
%{
~ get premise training data function
~ max 997 data pairs
~ format: [x,y,z,target]
~ data_pair_info: define how many data in one pair
~ num: how many data pair for the output
~ if num> 997, return -1 or return data pairs
%}
function premise_training_data = get_premise_training_data(num)
    %get the 1000 raw data 
    fileID = fopen('macky_data.txt','r');
    formatSpec = '%f';
    data_raw = fscanf(fileID,formatSpec);
    data_pair_base = 4; 
    
    %make the data pair
    [raw_data_num, column] = size(data_raw);    % 1000 raw data  
    data_pair_num = (raw_data_num - data_pair_base)+1;         %  997 data pairs
    data_mat_whole = ones(data_pair_num, data_pair_base);
    data_mat = ones(num, data_pair_base); 
        
    if num > data_pair_num
        fprintf("!!!Over max 997 data pairs!!! \n");
        premise_training_data = -1;
    else
        for i= 1: raw_data_num-3
            data_mat_whole(i,:) = [data_raw(i),data_raw(i+1),data_raw(i+2),data_raw(i+3)];
        end
        
        for i= 1: num
            data_mat(i,:) = data_mat_whole(i,:);
        end
    end
    
    premise_training_data = data_mat;
end

%*****************************layer5
%{
% layer 5 output function
%}
function O5 = layer5_output(O4)
    O5 = sum(O4);
end

%*****************************layer4
%{
 ~ layer 4 output function
 ~ calc the result of each rule into a vector
 ~ result format: wi*fi
 ~ data: training data, format[x,y,z,1], must be in a row
 ~ params: consqeuent params mat
%}
function O4 = layer4_output(O3, data, params, rule_num)
    output = ones(rule_num,1);
    
    for i = 1: rule_num
        output(i) = O3(i)*dot(params(:,i),data);
    end    
    
    O4= output;
end

%*****************************layer3
%{
 ~ layer 3 output function
 ~ calc the norm of fire strength and make them into a vector
 ~ norm formula: wi/sum(wi)
 ~ 
%}
function O3 = layer3_output(O2)
    sum_fire = sum(O2);
    [num_fire, column] = size(O2);
    output = ones(num_fire,1);
    for i =1: num_fire
        output(i) = O2(i)/sum_fire;
    end
    O3 = output;
end

%*****************************layer2
%{  
 ~layer 2 output function
 ~ mapping all the output of layer 1 to get all the firing strength into a
 vector
 ~ mapping is not universe, so input rule_num is changed and the mapping 
 algorithm has to be changed 
 ~ O1 is the layer1 output mat which must be transposed before put into O2
 function
 ~ O2 map all the rules to give "ONE" vector containing 8 firing strength
%}
function O2 = layer2_output(O1,mf_num_perinput, rule_num, input_num)
    output_mat = ones(rule_num, input_num);
    output_vector  = ones(rule_num,1);
  
    rr= 1; 
    rb= 1; 
    %make the combination matrix
    while rr<= rule_num 
        for i = 1: mf_num_perinput
            a = O1(1,i);
                for j = 1: mf_num_perinput
                    b = O1(2,j);
                        for k = 1: mf_num_perinput
                            c = O1(3,k);
                            output_mat(rr,:) = [a,b,c];
                            rr= rr+1;
                        end
                end
        end
    end
    
    %disp(output_mat);
    %{
    %get the fire strength vector by min
    min_ele = 100; 
    while rb< rule_num
        for i= 1: input_num
            if output_mat(rb,i)<min_ele
                    min_ele = output_mat(rb,i);    
            end
        end
        output_vector(rb) = min_ele;
        min_ele=100;
        rb = rb+1;
    end
    %}
    
    %get the fire strength vector by product
    product_ele =1; 
    while rb<= rule_num
        for i= 1: input_num
            product_ele = product_ele*output_mat(rb,i);
        end
        output_vector(rb) = product_ele;
        product_ele=1;
        rb = rb+1;
    end
    O2 = output_vector;
end

%*****************************layer1 
%{
 ~layer1 output function
 ~return a matrix containing all the output of 6 sigmoid mfs with 1 datapair
 ~format: [2*3], one column contains 2 calc result for one input, so 3
 columns data mat
%}
function O1= layer1_output(data_pair, mf_num_perinput, input_num, mf_params_mat, num_params_mf)
    output = ones(input_num, mf_num_perinput);
    params_marker =1; 
    
    for i = 1: input_num
        for j = 1: mf_num_perinput
                 params_mf_perinput = mf_params_mat(1:num_params_mf, params_marker: params_marker+1);
                 output(i,j) = sigmf(data_pair(i), params_mf_perinput(:,j));
        end
        params_marker = params_marker +2; 
    end
    
    O1 = output';
end


























