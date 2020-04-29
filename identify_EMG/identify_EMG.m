clear;

% 4試行で得られたデータそれぞれに対して信号処理を行う
[t1,ch_dt1(:,1),ch_dt1(:,2),ch_dt1(:,3),ch_dt1(:,4),sum1] = signal_processing("14_32_40check.txt","14_32_40para.txt");
[t2,ch_dt2(:,1),ch_dt2(:,2),ch_dt2(:,3),ch_dt2(:,4),sum2] = signal_processing("14_33_14check.txt","14_33_14para.txt");
[t3,ch_dt3(:,1),ch_dt3(:,2),ch_dt3(:,3),ch_dt3(:,4),sum3] = signal_processing("14_33_48check.txt","14_33_48para.txt");
[t4,ch_dt4(:,1),ch_dt4(:,2),ch_dt4(:,3),ch_dt4(:,4),sum4] = signal_processing("14_34_20check.txt","14_34_20para.txt");

dtsize(1) = size(t1,1); %データのサイズ
dtsize(2) = size(t2,1);
dtsize(3) = size(t3,1);
dtsize(4) = size(t4,1);

%動作判定とラベル付け
[lb_dt1(:,1),lb_dt1(:,2),lb_dt1(:,3),lb_dt1(:,4),dtsize2(1)] = label(sum1,dtsize(1));
[lb_dt2(:,1),lb_dt2(:,2),lb_dt2(:,3),lb_dt2(:,4),dtsize2(2)] = label(sum2,dtsize(2));
[lb_dt3(:,1),lb_dt3(:,2),lb_dt3(:,3),lb_dt3(:,4),dtsize2(3)] = label(sum3,dtsize(3));
[lb_dt4(:,1),lb_dt4(:,2),lb_dt4(:,3),lb_dt4(:,4),dtsize2(4)] = label(sum4,dtsize(4));
%ラベルデータをファイル出力
writematrix(lb_dt1,"labeldt1.csv");
writematrix(lb_dt2,"labeldt2.csv");
writematrix(lb_dt3,"labeldt3.csv");
writematrix(lb_dt4,"labeldt4.csv");

%パターン情報の取得
pt_dt1 = get_pattern(ch_dt1(:,1),ch_dt1(:,2),ch_dt1(:,3),ch_dt1(:,4),dtsize(1));
pt_dt2 = get_pattern(ch_dt2(:,1),ch_dt2(:,2),ch_dt2(:,3),ch_dt2(:,4),dtsize(2));
pt_dt3 = get_pattern(ch_dt3(:,1),ch_dt3(:,2),ch_dt3(:,3),ch_dt3(:,4),dtsize(3));
pt_dt4 = get_pattern(ch_dt4(:,1),ch_dt4(:,2),ch_dt4(:,3),ch_dt4(:,4),dtsize(4));
%パターン情報をファイル出力
writematrix(pt_dt1,"patterndt1.csv");
writematrix(pt_dt2,"patterndt2.csv");
writematrix(pt_dt3,"patterndt3.csv");
writematrix(pt_dt4,"patterndt4.csv");


%以下は関数の定義
%生筋電データに対して信号処理を行い，%MVCを得る．
function [t,ch1,ch2,ch3,ch4,sum] = signal_processing(filename,filename_para)
%ファイルの読み込み
data = readmatrix(filename,'NumHeaderLines',1,'Delimiter','\t'); %筋電データ
paradt = readmatrix(filename_para,'Delimiter','\t'); %各パラメータ
t_max = data(end,1);        %測定時間
ch1 = data(1:end,2);
ch2 = data(1:end,3);
ch3 = data(1:end,4);
ch4 = data(1:end,5);
fs = paradt(2,2);           %サンプリング周波数
t = data(1:end,1) / fs;     %時間
dtsize = size(t,1);         %データのサイズ
sum = zeros([dtsize 1]);    %各チャネルの正規化後の値の和
ch1_offset = paradt(3,2);   %ch1のオフセット
ch2_offset = paradt(4,2);   %ch2のオフセット
ch3_offset = paradt(5,2);   %ch3のオフセット
ch4_offset = paradt(6,2);   %ch4のオフセット
ch1_max = paradt(7,2);      %ch1の最大値
ch2_max = paradt(8,2);      %ch2の最大値
ch3_max = paradt(9,2);      %ch3の最大値
ch4_max = paradt(10,2);     %ch4の最大値
f = t / t_max * fs;         %周波数
deg = 2;                    %フィルタの次数
fc_low = 1.5;               %バンドパスフィルタの低域カットオフ周波数
fc_high = 100;              %バンドパスフィルタの高域カットオフ周波数
fc_low2 = 1;                %ローパスフィルタのカットオフ周波数

%バンドパス(1.5～100Hz)
[b,a] = butter(deg,[fc_low fc_high]/(fs/2),'bandpass');
ch1 = filter(b,a,ch1);
ch2 = filter(b,a,ch2);
ch3 = filter(b,a,ch3);
ch4 = filter(b,a,ch4);

%全波整流
ch1 = abs(ch1);
ch2 = abs(ch2);
ch3 = abs(ch3);
ch4 = abs(ch4);

%ローパス(1Hz)
[b, a] = butter(deg, fc_low2/(fs/2));
ch1 = filter(b,a,ch1);
ch2 = filter(b,a,ch2);
ch3 = filter(b,a,ch3);
ch4 = filter(b,a,ch4);

%オフセット除去
ch1 = ch1 - ch1_offset;
ch2 = ch2 - ch2_offset;
ch3 = ch3 - ch3_offset;
ch4 = ch4 - ch4_offset;
%負になったデータは0にする
for i = 1:dtsize
    if ch1(i) < 0
        ch1(i) = 0;
    end
    if ch2(i) < 0
        ch2(i) = 0;
    end
    if ch3(i) < 0
        ch3(i) = 0;
    end
    if ch4(i) < 0
        ch4(i) = 0;
    end
end

%正規化
ch1 = ch1 / ch1_max;
ch2 = ch2 / ch2_max;
ch3 = ch3 / ch3_max;
ch4 = ch4 / ch4_max;
%1を超えたデータは1にし，総和を計算
for i = 1:dtsize
    if ch1(i) > 1
        ch1(i) = 1;
    end
    if ch2(i) > 1
        ch2(i) = 1;
    end
    if ch3(i) > 1
        ch3(i) = 1;
    end
    if ch4(i) > 1
        ch4(i) = 1;
    end
    sum(i) = ch1(i) + ch2(i) + ch3(i) + ch4(i);
end

end


% %MVCからパターン情報を得る関数
% ch1～ch4には正規化した後のデータを入れる
function pt = get_pattern(ch1,ch2,ch3,ch4,dtsize)

sum = zeros(dtsize,1);
flag = false;
for i = 1:dtsize
    %各時間におけるチャンネルの総和を1に調整
    sum(i) = ch1(i) + ch2(i) + ch3(i) + ch4(i);
    if sum(i) == 0
        ch1(i) = 0;
        ch2(i) = 0;
        ch3(i) = 0;
        ch4(i) = 0;
    else
        ch1(i) = ch1(i) / sum(i);
        ch2(i) = ch2(i) / sum(i);
        ch3(i) = ch3(i) / sum(i);
        ch4(i) = ch4(i) / sum(i);
        
        %動作判定
        if sum(i) >= 0.5
            sum(i) = 1; 
        else
            sum(i) = 0;
        end
    end
    
    %動作部分のみを取り出し
    ch1(i) = ch1(i) * sum(i);
    ch2(i) = ch2(i) * sum(i);
    ch3(i) = ch3(i) * sum(i);
    ch4(i) = ch4(i) * sum(i);
    
    %安静状態はカット
    if sum(i) > 0
       if flag == false
           pt(1,1) = ch1(i);
           pt(1,2) = ch2(i);
           pt(1,3) = ch3(i);
           pt(1,4) = ch4(i);
           flag = true;
       else
           pt = vertcat(pt,[ch1(i) ch2(i) ch3(i) ch4(i)]);
       end
    end
end

end


%閾値よりも大きいデータのみを取り出し，ラベル付けを行う
%label1: 背屈，label2: 掌屈，label3: 尺屈，label4: 撓屈
function [label1,label2,label3,label4,dtsize2] = label(sum,dtsize)
for i = 1:dtsize
    if sum(i) >= 0.5
        sum(i) = 1;
    else
        sum(i) = 0;
    end
end

dtsize2 = dtsize;
for i = 1:dtsize2
    while sum(i) == 0
       sum(i) = [];
       dtsize2 = dtsize2 - 1;
    end
    label1(i) = sum(i);
    label2(i) = 0;
    label3(i) = 0;
    label4(i) = 0;
    if sum(i+1) == 0
        tmp = i + 1;
        break;
    end
end
for i = tmp:dtsize2
   while sum(i) == 0
       sum(i) = [];
       dtsize2 = dtsize2 - 1;
    end
    label2(i) = sum(i);
    label1(i) = 0;
    label3(i) = 0;
    label4(i) = 0;
    if sum(i+1) == 0
        tmp = i + 1;
        break;
    end
end
for i = tmp:dtsize2
   while sum(i) == 0
       sum(i) = [];
       dtsize2 = dtsize2 - 1;
    end
    label3(i) = sum(i);
    label1(i) = 0;
    label2(i) = 0;
    label4(i) = 0;
    if sum(i+1) == 0
        tmp = i + 1;
        break;
    end
end
for i = tmp:dtsize2
   while sum(i) == 0
       sum(i) = [];
       dtsize2 = dtsize2 - 1;
    end
    label4(i) = sum(i);
    label1(i) = 0;
    label2(i) = 0;
    label3(i) = 0;
    if sum(i+1) == 0
        tmp = i + 1;
        break;
    end
end
dtsize_tmp = dtsize2;
for i = tmp:dtsize_tmp
   dtsize2 = dtsize2 - 1; 
end

end