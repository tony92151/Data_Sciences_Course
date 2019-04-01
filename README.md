# Data_Sciences_Course

# HW1

>使用 jupyter notebook，需要GPU，可以從
https://colab.research.google.com/drive/1thciKmFzbOffnqvBYot7poUptIBxVerN


> 使用 python3 app.py，不需要GPU

使用colab執行

資料集：僅使用[台灣電力公司_未來一週電力供需預測](https://data.gov.tw/dataset/33462)之 尖峰負載(MW) 作為資料集

 ## Step1
 將 2017/1/1 ~ 2019/2/28 的資料切成七份分別為週一到週日之資訊

## Step2
將七份資訊去除異常部分，

> 當 A[ i ]與 A[ i+1 ]相差大於4000，則A[ i+1 ] = ( A[ i ] + A[ i+1 ] ) / 2

## Step3
使用正交化函式將數據正交化

## Step4

Convert a Time Series to a Supervised Learning Problem

將train_x 設為2組輸入，假如要預測今天(星期一)之資訊，輸入必須為上週星期一之用電量以及上上週星期一之用電量

[Reference1](https://machinelearningmastery.com/convert-time-series-supervised-learning-problem-python/)
​
[Reference2](https://blog.csdn.net/baidu_36669549/article/details/85595807)
​
## Step5
定義 LSTM 

## Step6
開始訓練，共七次，每次 10000 Epoch 


## Step7
定義 appendPrediction 函式用來將預測值加入原有數據，並可以重複迭代

## Step8

因數據末端到2019/2/28，所以估計append 10 週可以到4/8

## Step9

設 4/2~4/8 之 datetime 並使用 getPrediction 從預測值數列中找出預測值

## Step10

將預測值存成 submission.csv

<img src="https://github.com/tony92151/Data_Sciences_Course/blob/master/image/image2.png"/>

# Result

<img src="https://github.com/tony92151/Data_Sciences_Course/blob/master/image/image3.png"/>