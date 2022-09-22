clear;
clc;
tic
filenames = dir('./ICBHI_final_database/');
for i = 1:1845
    file{i,1} = filenames(i).name;
end
for i = 1:1843
    file_txt_wav{i,1} = file{i+2,1};
end
file_txt_wav = char(file_txt_wav);
files_txt = [];
files_wav = [];
step_txt = 1;
step_wav = 1;
for i =1:1843
    if strcmp(file_txt_wav(i,24:26),'txt')
        files_txt{step_txt,1} = file_txt_wav(i,:);
        step_txt = step_txt + 1;
    elseif strcmp(file_txt_wav(i,24:26),'wav')
        files_wav{step_wav,1} = file_txt_wav(i,:);
        step_wav = step_wav + 1;
    end
end
clear file file_txt_wav filenames step_txt step_wav
files_txt = char(files_txt);
files_wav = char(files_wav);
[file_len,~] = size(files_txt);
maindir = 'E:\1 科研项目与竞赛\2022大创\Respiratory_sound_recognition\';
datadir = 'E:\1 科研项目与竞赛\2022大创\Respiratory_sound_recognition\ICBHI_final_database\';

start_num = 131;
end_num = 150;
for j = start_num : end_num
    readfile_txt = strcat(datadir,files_txt(j,:));
    readfile_wav = strcat(datadir,files_wav(j,:));
    [data, fs] = audioread(readfile_wav);
    data = resample(data,4000,fs);

    txt_content = importdata(readfile_txt);
    len_txt = length(txt_content(:,1));
    for i = 1:len_txt
        sTime = txt_content(i, 1);
        eTime = txt_content(i, 2);
        timeLimits = [sTime eTime];  %循环开始时间
        frequencyLimits = [0 2000];  %循环结束时间
        overlapPercent = 50;
        data_ROI = data(:);
        sampleRate = 4000; % Hz
        startTime = 0; % seconds
        timeValues = startTime + (0:length(data_ROI)-1).'/sampleRate;
        minIdx = timeValues >= timeLimits(1);
        maxIdx = timeValues <= timeLimits(2);
        data_ROI = data_ROI(minIdx&maxIdx);
        timeValues = timeValues(minIdx&maxIdx);
        if txt_content(i,3)==0&txt_content(1,4)==0
            figure('visible','off')
            pspectrum(data_ROI,timeValues, ...
            'spectrogram', ...
            'FrequencyLimits',frequencyLimits, ...
            'OverlapPercent',overlapPercent);
            axis off
            delete(get(gca,'title'))
            colorbar('off')
            saveas(gcf, [maindir,'img\','txt00_', num2str(j),'_',num2str(i), '.jpg']);
        end

        if txt_content(i,3)==0&txt_content(1,4)==1
            figure('visible','off')
            pspectrum(data_ROI,timeValues, ...
            'spectrogram', ...
            'FrequencyLimits',frequencyLimits, ...
            'OverlapPercent',overlapPercent);  
            axis off
            delete(get(gca,'title'))
            colorbar('off')
            saveas(gcf, [maindir,'img\','txt01_', num2str(j),'_',num2str(i), '.jpg']);
        end

        if txt_content(i,3)==1&txt_content(1,4)==0
            figure('visible','off')
            pspectrum(data_ROI,timeValues, ...
            'spectrogram', ...
            'FrequencyLimits',frequencyLimits, ...
            'OverlapPercent',overlapPercent);
            axis off
            delete(get(gca,'title'))
            colorbar('off')
            saveas(gcf, [maindir,'img\','txt10_', num2str(j),'_',num2str(i), '.jpg']);
        end    

        if txt_content(i,3)==1&txt_content(1,4)==1
            figure('visible','off')
            pspectrum(data_ROI,timeValues, ...
            'spectrogram', ...
            'FrequencyLimits',frequencyLimits, ...
            'OverlapPercent',overlapPercent); 
            axis off
            delete(get(gca,'title'))
            colorbar('off')
            saveas(gcf, [maindir,'img\','txt11_', num2str(j),'_',num2str(i), '.jpg']);
        end

    end
end
toc