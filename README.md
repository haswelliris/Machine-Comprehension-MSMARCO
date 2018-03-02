# Machine-Comprehension-MSMARCO
# R-NET reimplemention
## 使用简介
### 预先准备训练数据
自行复制训练所需数据集和单词编码集
```
vocabs.pkl
glove.6B.100d.txt dev.ctf
dev.ctf
test.ctf
train.ctf
```
到script目录下
### 训练
如果在天河则执行`source ~/hch/soft/pre_env.sh`加载环境配置  
在dell上执行`source /home/u1395/soft/my_pre_env.sh`加载配置。
进入script文件夹，执行命令
```
# 单卡训练
python train_rnet.py --logdir ./logs/

# 多卡训练 请把4改成显卡数量
 mpirun -npernode 4 python train_rnet.py --logdir ./logs/
```
### 测试
使用训练好的模型求解dev数据集
```
python train_rnet.py --test dev.tsv
```
得到结果`pm.model_out.json`，将该文件移动到 `../ms_marco_eval` 文件夹  
进入ms_marco_eval文件夹，运行命令
```
sh ./run.sh dev_as_references.json pm.model_out.json
```
得到类似如下输出，就是最终结果
```
{'testlen': 108948, 'reflen': 111890, 'guess': [108948, 103691, 99301, 94994], 'correct': [41880, 22693, 18713, 16777]}
ratio: 0.9737063187058631
############################
bleu_1: 0.3741621473845382
bleu_2: 0.28231995043944164
bleu_3: 0.24452041494674448
bleu_4: 0.2239025217932479
rouge_l: 0.302060927811
############################
```