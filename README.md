# Machine-Comprehension-MSMARCO
# origin demo
## Bi-directional Attention Flow
clone from `https://github.com/Microsoft/CNTK/tree/nikosk/bidaf/Examples/Text/BidirectionalAttentionFlow/msmarco/team_xyz`  
该demo的原论文链接:`https://arxiv.org/abs/1611.01603`  
论文分析稍后给给出，该分支为对该代码的改进，及参数调整，以适合我们的机器环境
## 使用简介
### 预先准备训练数据
自行复制训练所需数据集和单词编码集，为了更好的效果，最好不要使用原代码给出的词向量和字符向量训练
```
词向量
wget http://nlp.stanford.edu/data/glove.840B.300d.zip
字符向量
https://github.com/minimaxir/char-embeddings
转化
convert_msmarco.py
ans_query2ctf.py (seq2seq模型训练用)
tsv2ctf.py
```
到script目录下
### 训练
如果在天河则执行`source ~/hch/soft/pre_env.sh`加载环境配置  
在dell上执行`source /home/u1395/soft/my_pre_env.sh`加载配置。
进入script文件夹，执行命令
```
# 单卡训练
python train_pm.py --logfile log1

# 多卡训练 请把4改成显卡数量
 mpirun -npernode 4 python train_pm.py --logfile log1
>>>>>> origin/r-net
```
### 测试
使用训练好的模型求解dev数据集
```
python train_pm.py --test dev.tsv
```
得到结果`pm.model_out.json`，将该文件移动到 `../ms_marco_eval` 文件夹  
进入ms_marco_eval文件夹，运行命令
```
sh ./run.sh dev_as_references.json pm.model_out.json
```
得到类似如下输出，就是最终结果
```
{'testlen': 90258, 'reflen': 102237, 'guess': [90258, 85848, 81733, 77660], 'correct': [32149,
 16430, 13472, 12054]}
ratio: 0.8828310689867573
############################ 
bleu_1: 0.3119194095480127
bleu_2: 0.22864157835651136
bleu_3: 0.1961410805365376
bleu_4: 0.1789575527634363
rouge_l: 0.283029872284
############################ 
```
