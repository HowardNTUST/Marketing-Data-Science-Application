#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  3 16:04:58 2018

@author: Howard Chung
"""


# Imports the Google Cloud client library
from google.cloud import speech
from google.cloud.speech import enums
from google.cloud.speech import types
from google.cloud import storage
import datetime
import io
import os
import time
import numpy as np
import pandas as pd

'''
# 語音基礎串接
一分鐘內語音辨識
'''

os.chdir('your working directory')
#os.chdir('/home/slave1/git/speech2text_1min')

def speech_to_text_in_a_min(title_pattern='nlpno', 
                            wd ='re',
                            json_os = 'speech2text-3de4444fd46a.json',
                            sample_rate_hertz =  48000):
    '''
    * json_os：憑證檔的路徑
    * title_pattern：錄音檔的名稱模式
    * sample_rate_hertz：錄音的取樣頻率
    * doc_title：docx文件名稱
    * wd：工作目錄
    
    '''
    
    # 計時
#    start_time = time.time()
    # 從python client端對雲端speech2text服務進行驗證
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] =json_os
    client = speech.SpeechClient()
    
    os.chdir(wd)
    file_list = os.listdir()     
    
    # 選出title_pattern的錄音檔
    select_wav = []
    for i in file_list:
        if title_pattern in i:
            select_wav.append(i)
         
        # [START migration_sync_request]
        # [START migration_audio_config_file]
    
    aa = pd.DataFrame()
    
    for music in select_wav:
            
        # 將 audio錄音檔 讀入進來
        with io.open(music, 'rb') as audio_file:
            content = audio_file.read()
        
        # 將錄音檔轉換成google 看得懂的格式
        audio = types.RecognitionAudio(content=content)
        
        # 設定格式錄音檔
        config = types.RecognitionConfig(
             encoding=enums.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=sample_rate_hertz,
            language_code='cmn-Hant-TW' ,
            enable_word_time_offsets=True)
        
        # 機器學習文字辨識(speech2text)
        print('')
        response = client.recognize(config, audio)
            
        
        transcript_list = []
        transcript_confidence = []
        timerecored = []
        # Each result is for a consecutive portion of the audio. Iterate through
        # them to get the transcripts for the entire audio file.
        for result in response.results:
            alternative = result.alternatives[0]
            # The first alternative is the most likely one for this portion.
            transcript_list.append(alternative.transcript)
            transcript_confidence.append(alternative.confidence)
            print('Transcript: {}'.format(alternative.transcript))
            print('Confidence: {}'.format(alternative.confidence))
            
            
            # begining and end time of a sentence
            sentence_start_time = alternative.words[0].start_time
            sentence_end_time = alternative.words[len(alternative.words)-1].end_time
            
            # make time
            sentence_start_time = round( sentence_start_time.seconds + sentence_start_time.nanos * 1e-9)
            sentence_end_time = round( sentence_end_time.seconds + sentence_end_time.nanos * 1e-9)
            
            # make min
            sentence_start_time= str(datetime.timedelta(seconds=sentence_start_time))
            sentence_end_time =str(datetime.timedelta(seconds=sentence_end_time))
            timerecored.append([sentence_start_time, sentence_end_time])
            
        # pandas 建立信心程度資料表
         # make df
        transcript_df = pd.DataFrame(transcript_list, columns = ['文章段句'])
        confidence_df = pd.DataFrame(transcript_confidence, columns = ['機器認字信心水準'])
        confidence_df['機器認字信心水準'] = round(confidence_df['機器認字信心水準'],2)
        time_df  = pd.DataFrame(timerecored, columns = ['start', 'end'])
        correctness_summary_df = pd.concat([transcript_df , confidence_df,time_df], axis = 1)    
        correctness_summary_df = correctness_summary_df.sort_values(['機器認字信心水準'])
        correctness_summary_df['改善順序'] = range(1, len(correctness_summary_df)+1)
        
        timer_translist =[]
        for hah,timer in zip(transcript_list,timerecored):
           timer_translist.append(hah+'  ' +'【'+' to '.join(timer)+'】')
        
        aa = pd.concat([ aa, correctness_summary_df])
    

    return aa.to_csv('文章認字信心矩陣.csv') 


# main
matr = speech_to_text_in_a_min(title_pattern='nlpno', 
                            wd ='re',
                            json_os = 'speech2text-3de4444fd46a.json',
                            sample_rate_hertz =  48000)

