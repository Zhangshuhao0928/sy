# SY will graduate successfully.
> *Just for sy*
> 
> *Just believe me and do ur best*
> 
> *Everything will be fine ～*

### 环境配置
1. 配置 Conda 环境:
    ```shell
    conda create -n sy python=3.8.10
    ```
2. 进入环境：
    ```shell 
    conda activate sy
    ```
3. 安装各种 python packages:
    ```shell
    pip install -r requirements.txt 
   ```
   等待安装完成即可，如果说有的 package 安装补上则添加源
    ```shell
    pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple
   ```


### 主函数文件
```
python sy_main.py
```

### 编码风格
示例：
```python
# 创建 metric list
def generate_metrics_list(metrics_def: Dict) -> Dict:
    """
    @param metrics_def: metric传入字典
    @return: 输出一个包含各种 key 但是对应 value 为空的字典
    """

    # 创建一个空字典，用于存储各指标的值列表
    result_dict = {}

    # 遍历输入的 metrics_def 字典中的指标名称
    for name in metrics_def.keys():
        # 将每个指标名称作为键，对应的值初始化为空列表
        result_dict[name] = []

    # 返回生成的字典
    return result_dict
```
 * *metrics_def: Dict：*
   * metrics_def 为参数命名， Dict 为该参数的类型，说明其是一个字典对象；
 * *-> Dict：*
   * -> 表示暗示输出， 说明该函数输出一个字典类型的对象作为返回值；
 * *@param：*
   * 说明了每个参数的含义，方便理解

### 终将胜利，别怕
![](pic/together.jpg)