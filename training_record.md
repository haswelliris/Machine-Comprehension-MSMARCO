# 训练记录
## 1.原始版本适配性修改——代号：源计划
### 更新细节
1、修改config.py和train_pm.py使之支持多显卡运行  
2、修改tsv2ctf.py，将9个字段的dev.tsv截取为只有4个字段(只有问题没有答案)的测试输入数据  
3、修改测试代码eval.py，增加对重复的问题id的判断、去掉检验数据(dev_as_references.json)中有但是测试数据(dev.tsv pm.model_out.json)中没有的问题id  
4、该版本完全备份v0.1.tar.gz  
5、更多细节详见./ms_marco_eval/0211_readme.txt
### 特性
1、多显卡并行训练  
2、成功完成从数据预处理到训练再到评估正确率的完整步骤  
3、训练参数是原始代码的，没有改动，速度大概为40 samples/s，整个训练时间长达两三天
### 最终评价
#### dev.tsv : 0.2773
input file: dev.tsv  
validated file: dev_as_references.json
```
sh ./run.sh dev_as_references.json pm.model_out.json
{'testlen': 93619, 'reflen': 102237, 'guess': [93619, 89209, 84882, 80578], 'correct': [32230, 16059, 13136, 11721]}
ratio: 0.9157056642898274
############################
bleu_1: 0.31399141886142895
bleu_2: 0.22705158552741816
bleu_3: 0.19377869574092701
bleu_4: 0.17626737413154606
rouge_l: 0.277343126374
############################
```
#### test.tsv : 0.4111
input file: test.tsv
validated file: output.json
```
sh ./run.sh output.json test.tsv_pm.model_out.json
{'testlen': 111437, 'reflen': 102016, 'guess': [111437, 106121, 100917, 95748], 'correct': [51931, 37970, 34447, 32172]}
ratio: 1.0923482590966016
############################
bleu_1: 0.46601218625770197
bleu_2: 0.40833656257599327
bleu_3: 0.384657794196928
bleu_4: 0.3718715807249601
rouge_l: 0.411136705716
############################
```
## 2.第一次参数改进——代号：一小步
#### dev.tsv : 0.2518
input file: dev.tsv(0212_dev.json)  
validated file: dev_as_references.json
```
./run.sh dev_as_references.json 0212_dev.json
{'testlen': 94087, 'reflen': 102237, 'guess': [94087, 89677, 85267, 80862], 'correct': [29962,13985, 11076, 9741]}
ratio: 0.9202832633977823
############################
bleu_1: 0.2920261487166789
bleu_2: 0.2043580977406728
bleu_3: 0.17070860354986164
bleu_4: 0.15310926230045352
rouge_l: 0.251853089966
############################
```
## 3.第二次参数改进——代号：进退维谷
input file: dev.tsv(0213_dev.json)  
validated file: dev_as_references.json
#### dev.tsv : 0.2570
```
sh ./run.sh dev_as_references.json 0213_dev.json
{'testlen': 91289, 'reflen': 102237, 'guess': [91289, 86879, 82470, 78062], 'correct': [30288, 14378, 11481, 10115]}
ratio: 0.8929154806968035
############################
bleu_1: 0.2942853130306784
bleu_2: 0.20784249525401285
bleu_3: 0.17472549775505625
bleu_4: 0.1573540329902145
rouge_l: 0.257043369997
############################ 
```
## 4.第三次参数改进——代号：K80
### 更新细节
1、 'minibatch_size'    : 32768  
2、 'epoch_size'        : 164650  
3、 'log_freq'          : 1000  
4、 'minibatch_seqs'    : 24  
### 特性
增大batch_size，缩小batch_seqs，使整体显存使用量达到10GB以上，训练时显存使用情况基本在11000MiB/11439MiB，利用很充分，效率很高,能达到95 samples/s。  
每次迭代显存使用量都会上涨，最终因爆显存训练被迫中断。。。  
所以还需要继续改进。
```
Finished Epoch[56 of 300]: [Training] loss = 7.178536 * 164736, metric = 0.00% * 164736 1738.402s ( 94.8 samples/s);
Finished Epoch[56 of 300]: [Training] loss = 7.178536 * 164736, metric = 0.00% * 164736 1738.402s ( 94.8 samples/s);
Finished Epoch[56 of 300]: [Training] loss = 7.178536 * 164736, metric = 0.00% * 164736 1738.402s ( 94.8 samples/s);
Finished Epoch[56 of 300]: [Training] loss = 7.178536 * 164736, metric = 0.00% * 164736 1738.402s ( 94.8 samples/s);
Validated 5407 sequences, loss 7.0166, F1 0.1211, EM 0.0520, precision 0.128213, recall 0.133399 hasOverlap 0.152580, start_match 0.066950, end_match 0.107823
Validated 5407 sequences, loss 7.0166, F1 0.1211, EM 0.0520, precision 0.128213, recall 0.133399 hasOverlap 0.152580, start_match 0.066950, end_match 0.107823
Validated 5407 sequences, loss 7.0166, F1 0.1211, EM 0.0520, precision 0.128213, recall 0.133399 hasOverlap 0.152580, start_match 0.066950, end_match 0.107823
Validated 5407 sequences, loss 7.0166, F1 0.1211, EM 0.0520, precision 0.128213, recall 0.133399 hasOverlap 0.152580, start_match 0.066950, end_match 0.107823
CUDA failure 2: out of memory ; GPU=0 ; hostname=gn26 ; expr=cudaMalloc((void**) &deviceBufferPtr, sizeof(AllocatedElemType) * AsMultipleOf(numElements, 2))
^C^CAbort is in progress...hit ctrl-c again within 5 seconds to forcibly terminate

```
### 最终评价
#### dev.tsv : 0.2743
input file: dev.tsv(0215_dev.json)  
validated file: dev_as_references.json
```
sh ./run.sh dev_as_references.json 0215_dev.json
{'testlen': 96940, 'reflen': 102237, 'guess': [96940, 92530, 88182, 83882], 'correct': [32608, 15908, 12844, 11400]}
ratio: 0.9481890118058927
############################
bleu_1: 0.31848604489896654
bleu_2: 0.2276912515104176
bleu_3: 0.1926463420756275
bleu_4: 0.17415939571650016
rouge_l: 0.2743135975
############################
```
#### test.tsv : 0.4104
input file: test.tsv(0215_test.json)
validated file: output.json
```
sh ./run.sh output.json 0215_test.json
{'testlen': 112607, 'reflen': 102016, 'guess': [112607, 107291, 102049, 96876], 'correct': [52533, 38251, 34686, 32455]}
ratio: 1.1038170483061371
############################
bleu_1: 0.46651629117194787
bleu_2: 0.407824354357397
bleu_3: 0.38379318644237065
bleu_4: 0.37097043474371627
rouge_l: 0.410479024469
############################
```
速度上去了，但效果不是很好，因为中途就终止了。
## 4.第四次改进——代号：以退为进
### 更新细节
1、修改保存函数，防止IO爆炸导致无法保存model而卡死  
2、缩小batch_seqs,epoch_size
```
    'minibatch_size'    : 24576,   #24576 8192 in samples when using ctf reader, per worker
    'epoch_size'        : 82325,   #82325 in sequences, when using ctf reader
    'log_freq'          : 2000,     #500 in minibatchs
    'max_epochs'        : 300,
    'lr'                : 2,
    'train_data'        : 'train.ctf',  # or 'train.tsv'
    'val_data'          : 'dev.ctf',
    'val_interval'      : 1,       # interval in epochs to run validation
    'stop_after'        : 2,       # num epochs to stop if no CV improvement
    'minibatch_seqs'    : 20,   
```
这次训练很成功，速度较快且loss一直在降低，150次时loss是6.62，直到300次时终止训练，最终的loss是6点零几，但很抱歉当时没有及时保存输出，所以具体是多少现在看不到了。
### 最终评价
#### test.tsv
input file: test.tsv(0219_test.json)
validated file: output.json
```
sh ./run.sh output.json 0219_test.json
{'testlen': 107605, 'reflen': 102016, 'guess': [107605, 102289, 97745, 93347], 'correct': [63475, 50016, 46024, 43271]}
ratio: 1.0547855238393873
############################
bleu_1: 0.5898889456809573
bleu_2: 0.5370628851553695
bleu_3: 0.5140200155438105
bleu_4: 0.5009093802137201
rouge_l: 0.559090126351
############################
```
成绩在0.559左右，非常好

#### dev.tsv : 0.2743
input file: dev.tsv(0219_dev.json)  
validated file: dev_as_references.json
```
ratio: 0.9330868472275112
############################
bleu_1: 0.3495912620451068
bleu_2: 0.25993214464571157
bleu_3: 0.2239684873650862
bleu_4: 0.20453447421813606
rouge_l: 0.313725403687
############################
```
dev下面就很一般，可能过拟合了。

## 5.第五次改进：终章
### 更新细节
1、加大minibatch_seqs以提高每次迭代的数据量，提高显存利用率，但不能增加过多，防止爆显存导致训练终止  
2、把最大训练次数修改修改为600，使得loss能继续下降
```
    'max_epochs'        : 600,
    'minibatch_seqs'    : 24,      #num sequences of minibatch, when using tsv reader, per worker
```
现在还没训练完,遇到IO爆炸，卡住不动了。。。
目前的状况是这样的：
```
Finished Epoch[248 of 600]: [Training] loss = 6.021252 * 82444, metric = 0.00% * 82444 1086.490s ( 75.9 samples/s);
Finished Epoch[248 of 600]: [Training] loss = 6.021252 * 82444, metric = 0.00% * 82444 1086.490s ( 75.9 samples/s);
Finished Epoch[248 of 600]: [Training] loss = 6.021252 * 82444, metric = 0.00% * 82444 1086.490s ( 75.9 samples/s);
Finished Epoch[248 of 600]: [Training] loss = 6.021252 * 82444, metric = 0.00% * 82444 1086.490s ( 75.9 samples/s);
Validated 5407 sequences, loss 6.2454, F1 0.1776, EM 0.0873, precision 0.188952, recall 0.185432 hasOverlap 0.212132, start_match 0.108378, end_match 0.155354
Validated 5407 sequences, loss 6.2454, F1 0.1776, EM 0.0873, precision 0.188952, recall 0.185432 hasOverlap 0.212132, start_match 0.108378, end_match 0.155354
Validated 5407 sequences, loss 6.2454, F1 0.1776, EM 0.0873, precision 0.188952, recall 0.185432 hasOverlap 0.212132, start_match 0.108378, end_match 0.155354
```
但是loss下降曲线非常缓慢，每次在0.01左右，应该差不多算是训练到位了。
用此时中途得到的model进行测试。
### 最终评价
#### test.tsv
input file: test.tsv(0224_test.json)
validated file: output.json
```
$ sh ./run.sh output.json 0224_test.json
{'testlen': 108970, 'reflen': 102016, 'guess': [108970, 103654, 99113, 94687], 'correct': [62020, 48099, 44076, 41354]}
ratio: 1.068165777917179
############################
bleu_1: 0.5691474717812189
bleu_2: 0.5139103875594211
bleu_3: 0.4897210608333264
bleu_4: 0.47590292911548604
rouge_l: 0.553013702372
############################
```
成绩在0.553左右，不如上一个版本，应该是没有训练完成导致精确度不够高的原因。

#### dev.tsv : 0.2743
input file: dev.tsv(0224_dev.json)  
validated file: dev_as_references.json
```
$ sh ./run.sh dev_as_references.json 0224_dev.json
{'testlen': 95131, 'reflen': 102237, 'guess': [95131, 90721, 86707, 82767], 'correct': [36268, 19894, 16572, 14900]}
ratio: 0.930494830638605
############################
bleu_1: 0.35380262562082504
bleu_2: 0.26832888218863793
bleu_3: 0.23374286950306764
bleu_4: 0.2149195480884845
rouge_l: 0.319693731523
############################

```
dev下面还是很一般。
