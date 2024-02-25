# 实验结果

### 文件结构
1. 程序名字：
   * 为了方便之后多进程跑程序，所以之前的结构不行，文件会被覆盖掉；
   * 这里采用了运行程序的当前时间戳 **time** 来作为程序命名，保证每一个进程的唯一性；
   * 示例： [2024-02-22 06-31-42](2024-02-22%2006-31-42)
2. 每个进程内的文件结构：
   * dat 文件夹：
     * 存储 **.dat** 文件，示例 [prediction_epoch10_dimension200.dat](2024-02-22%2006-31-42/dat/prediction_epoch10_dimension200.dat)
   * plot 文件夹：
     * 存储各种 **图片** 数据；
     * contrast 对比图： [contrast10.png](2024-02-22%2006-31-42/plot/contrast10.png);
     * speed contrast 对比图： [speed_contrast_epoch10_dimension200.png](2024-02-22%2006-31-42/plot/speed_contrast_epoch10_dimension200.png);
     * 绝对误差 error 对比图： [error_epoch10_dimension200.png](2024-02-22%2006-31-42/plot/error_epoch10_dimension200.png)
   * saved_model 文件夹：
     * 存储 **模型参数** 文件（.pt）；
     * 每间隔一定 epoch 存储当前模型参数，[saved_model_epoch10.pt](2024-02-22%2006-31-42/saved_models/saved_model_epoch10.pt)
     * 训练全部完成后，存储最优模型参数， [best_model_epoch10.pt](2024-02-22%2006-31-42/saved_models/best_model_epoch10.pt)
   * 训练实时全参数对比图，方便追踪实验性能： ![all_curves.png](2024-02-22%2006-31-42/all_curves.png)
   * 训练完成后存储全部实验指标，即保存上图的各种参数： [curves.csv](2024-02-22%2006-31-42/curves.csv)
   * 训练完成后存储最优实验指标： [best_config.csv](2024-02-22%2006-31-42/best_config.csv)

以上，保证了只要在算力足够的情况表下，可以并行分布式跑大量的实验，从而快速验证效果，***不用担心找不到实验结果～ ^ ^***

<video width="700" height="400" controls>
    <source src="../pic/fire.mp4" type="video/mp4">
</video>