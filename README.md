# nre
Neural Relation Extraction (NRE) with wechat API

SemEval2010_task8 relation extraction

use cnn
keras TensorFlow Backend

# 1. 关系中文
中文识别直接采用keras_input_zh脚本执行

## 关系类型：

>部分-整体(e1,e2)  
部分-整体(e2,e1)  
内容-容器(e1,e2)  
内容-容器(e2,e1)  
产品-生产者(e1,e2)  
产品-生产者(e2,e1)  
成员-组织(e1,e2)  
成员-组织(e2,e1)  
工具-代理(e1,e2)  
工具-代理(e2,e1)  
起因-影响(e1,e2)  
起因-影响(e2,e1)  
消息-话题(e1,e2)  
消息-话题(e2,e1)  
无

## 训练数据样例：

>工具-代理(e2,e1)|\<per>你</per>这\<instrument>招</instrument>打得很不错  
部分-整体(e2,e1)|\<org>微软公司</org>的\<org>财务科</org>  
部分-整体(e1,e2)|\<loc>钓鱼岛</loc>是\<loc>中国</loc>的领土  
内容-容器(e2,e1)|\<weapon>导弹</weapon>携带的\<chemical>炸药</chemical>  
工具-代理(e2,e1)|\<per>阿拉法特</per>乘\<instrument>飞机</instrument>抵达巴黎  
工具-代理(e2,e1)|\<per>雷军</per>乘\<instrument>出租车</instrument>回家  
工具-代理(e2,e1)|\<per>牙医</per>使用\<instrument>电钻</instrument>对牙齿进行修复  
起因-影响(e2,e1)|最近\<disease>流感</disease>\<bio>病毒</bio>爆发  
起因-影响(e1,e2)|吸\<chemical>冰毒</chemical>可能导致\<disease>死亡</disease>  

通过标签将实体区分出来。通过|分割关系类型和训练语料  
标签说明： 人物：\<per>  组织：\<org>  地理位置：\<loc> 等等   
测试集为右侧数据，如：  
> \<per>你\</per>准备坐\<instrument>船</instrument>去那边  
结果为左侧分类数据  
> 工具-代理(e2,e1)

接着通过关系词词典筛选具体的关系类型。

1. 实体分类
	能否做到多分类，多分类的效果如何保证。
2. 分类体系：
	通用和要求是否有区别，能否做到全覆盖。
3. 具体关系：
	关系词词典构建->选取所有仅仅包含2个实体的句子
	1. 统计句子中的除实体外的词频（）这里可以采用某种聚类
	2. 对句子进行句法分析，选择包含两个实体的最小的句法树，统计句法树中的名词，动词词频
	3. 对前两步的结果进行综合，获取关系词词典
