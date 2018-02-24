用的是0209开始训练，0211终止训练的模型(未修改训练参数和网络结构，只解决了训练环境下多卡兼容问题)
using the model started training at 0209 and ended an 0211(not change the params and network framework,just fix the multi gpu training problems)
修改tsv2ctf.py，将9个字段的dev.tsv截取为只有4个字段(只有问题没有答案)的测试输入数据
change tsv2ctf.py, convert 9 params data of dev.tsv to 4 params data which can be the input of the test stage
修改测试代码eval.py，增加对重复的问题id的判断、去掉检验数据(dev_as_references.json)中有但是测试数据(dev.tsv pm.model_out.json)中没有的问题id
change the ms_marco_eval.py, add query_id repetition judgement ,and delete the query_id which in dev_as_references.json but not in dev.tsv(pm.model_out.json)

训练好的模型保存在 ~/BIGDATA/team_zly/backup_model/0211
因为该版本为完整跑通的第一个版本，故做了完全备份，位于 ~/BIGDATA/backup/v0.1
对于修改后的eval代码运行结果能否采用，需再商议。
对于测试数据的query_id重复问题，需请教老师同学。

测试结果如下：
test result:
1.
input file: dev.tsv
validated file: dev_as_references.json

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

2.
input file: test.tsv
validated file: output.json
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

