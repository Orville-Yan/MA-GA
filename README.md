1. 项目推进见飞书文档 https://x1rzpzd8gc3.feishu.cn/wiki/Flozw2hg9iXrzKkf1JLc3YQ1nKc
2. Notion由于空间问题，以后不用了，算子池也已经转移到飞书文档里面去了，大家注意查看。
3. OLDCODE里面加了G3GP这个文件，是遗传算法的主要框架，大家可以从这里面看如何添加Terminal和Primitives。
## 关于数据读取模块
- 主要的数据读取功能在DataReader里面，分成Barra/MinuteData/DayData三部分
- 需要调整的东西是文件路径，在Data_tools里面，分别是
  - root_path：整个日频文件夹的位置
  - barra_path/dict_path：barra和dict文件所在位置
  - minute_data_path：分钟频数据所在位置
