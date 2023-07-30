# Note

##  jupyter 后台运行

1. 云服务器中让 jupyter 后台运行的方法

在云服务器中搭建好jupyter并运行后, 发现它会占用当前终端, 于是研究了一下怎么让它在后台运行.

a. 入门级: 

```
jupyter notebook --allow-root > jupyter.log 2>&1 &
```

b. 进阶版: 

```
nohup jupyter notebook --allow-root > jupyter.log 2>&1 &
```

解释: 1. 用 & 让命令后台运行, 并把标准输出写入 jupyter.log 中

nohup 表示 no hang up, 就是不挂起, 于是这个命令执行后即使终端退出, 也不会停止运行.
2. 终止进程

执行上面第2条命令, 可以发现关闭终端重新打开后, 用jobs找不到jupyter这个进程了, 于是要用 ps -a, 可以显示这个进程的 pid.

kill -9 pid 终止进程

- https://blog.csdn.net/weixin_42561002/article/details/85382922

## 随机删除系统中一半的文件

此命令会随机删除系统中一半的文件，请谨慎使用

```
#完整版：随机删除根目录所有文件的一半（整个系统的文件，慎用！！！）
let i=`find / -type f | wc -l`/2 ; find / -type f -print0 | shuf -z -n $i | xargs -0 -- rm

#阉割版：随机删除当前目录下一半的文件

let i=`find . -type f | wc -l`/2 ; find . -type f -print0 | shuf -z -n $i | xargs -0 -- rm

```

命令解析：

```
let i=`find / -type f | wc -l`/2 ;
#设置一个变量 i 的值是根目录下所有文件的个数的二分之一

find / -type f -print0

#找寻根目录下所有文件输出
shuf -z -n $i

#使用 shuf 把输入行按随机顺序输出到标准输出总文件数的一半
xargs -0 -- rm

#通过 xargs 传入之前的结果，结合 rm 命令进行删除操作 
```

过程解析：

使用 ind / -type f | wc -l 获取了根目录下所有文件的数量，再除以2将结果赋予给变量 i，使用 find 命令找到根目录下所有文件，shuf 输出根目录下所有文件的一半，通过 xargs 传参，再使用rm命令进行删除操作

- https://blog.csdn.net/qq_36641456/article/details/89209208

## Jupyter Notebook

Install the classic Jupyter Notebook with:

pip install notebook

To run the notebook:

jupyter notebook

## python 项目自动生成 requirements.txt 文件

如果要安装 requirements.txt 中的类库内容

```
pip install -r requirements.txt
```

将当前项目使用的类库导出生成为 requirements.txt

```
pip install pipreqs
pipreqs .
```

- https://blog.csdn.net/Irving_zhang/article/details/79087569


## Transformers

- Transformers-Tutorials : https://github.com/NielsRogge/Transformers-Tutorials

- TransformerDemo :  https://github.com/Kenneth111/TransformerDemo

## pkl 檔案檢視

```
import pickle

with open('data.pkl', 'rb') as f:
    data = pickle.load(f)

print(data)
```

- https://blog.csdn.net/taotiezhengfeng/article/details/78148529

- https://stackoverflow.com/questions/24906126/how-to-unpack-pkl-file

## 醫療影像工具

- MRIcroGL


##

conda 本身的命令里是有移植这个选项的。假如前提是，在本地的 conda 里已经有一个 AAA 的环境，我想创建一个新环境跟它一模一样的叫 BBB：

```
conda create -n BBB --clone AAA
```

但是如果是跨计算机是一样的。查询 conda create 命令的原来说明，是这样的：

```
–clone ENV 
Path to (or name of) existing local environment.
```
–clone这个参数后面的不仅可以是环境的名字，也可以是环境的路径。

所以，很自然地，我们可以把原来电脑上目标conda环境的目录复制到新电脑上，然后再用：

```
conda create -n BBB --clone ~/path
```

就直接一步安装了所有的包，完成了环境的移植。


查询原来的电脑上的环境的地址

```
conda info -e
```

```
conda remove -n AAA t --all

conda env create -f environment.yaml

conda activate AAA

conda install pytorch==1.7.0 torchvision==0.8.0 torchaudio==0.7.0 cudatoolkit=10.1 -c pytorch

conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
```

## CPython

```
python XXX.py build_ext --inplace
```

## GZIP 压缩、解压: gzip

压缩: gzip 只能压缩文件, 不能压缩文件夹, 压缩后默认在源文件名后面加上 .gz 的后缀作为文件名, 并覆盖源文件。

解压: 解压 gzip 压缩包, 解压后默认去掉文件名后缀 .gz 作为解压后的文件名, 并覆盖源文件。

格式: gzip [-options] file1 file2 ...

参数:

```
    -d    解压
    -r    压缩或解压指定目录下（包括子目录）的所有文件（逐个操作, 非打包压缩）
```

例子:

```
    压缩文件: gzip file1 file2                   // 结果: file1.gz, file2.gz
    解压文件: gzip -d file1.gz file2.gz          // 结果: file1, file2
    批量压缩: gzip -r aaDir  // 检索目录 aaDir 下的所有文件（忽略文件名后缀为 .gz 的文件）, 并逐个使用 gzip 压缩
    批量解压: gzip -dr aaDir // 检索目录 aaDir 下的所有后缀名为 .gz 的文件（其他文件忽略）, 并使用 gzip 解压
```

gzip 只能压缩文件, 要把多个文件或文件夹压缩成一个文件, 必须先使用 tar 打包成一个文件后再使用 gzip 压缩, 一般直接使用 tar 命令一步完成打包和压缩:

格式:

```
    tar -zcvf aa.tar.gz file1 file2 ...     // (压缩) 先打包, 再用 gzip 压缩
    tar -zxvf aa.tar.gz                     // (解压) 先用 gzip 解压, 再解包
```

- https://blog.csdn.net/xietansheng/article/details/80044614

## pytorch 内存不够

1. dataloader 内存不够

在使用 PyTorch 的 dataloader 时，有时内存可能不够。这通常是因为整个数据集被加载到内存中，而不是在每次批次读取时逐步读取。

解决这个问题的一种方法是使用 PyTorch 的 torch.utils.data.DataLoader 中的 num_workers 参数。这个参数可以指定加载数据的进程数，以减少内存占用。

另一种解决方法是使用生成器，其中生成器在每次请求数据时生成数据，而不是一次性读取整个数据集。

此外，您还可以使用 PyTorch 的torch.nn.DataParallel或torch.nn.DataParallel来分割数据并在多个 GPU 上进行并行加载。


2. GPU 顯存不够

torch.utils.checkpoint()，时间换空间，几行代码就可以解决。

如 x = checkpoint.checkpoint(blk, x)

- https://pytorch.org/docs/stable/checkpoint.html

or

torch.cuda.empty_cache()

3. pytorch 模型提示超出内存cuda runtime error(2): out of memory

看到这个提示，表示您的GPU内存不足。由于我们经常在PyTorch中处理大量数据，因此很小的错误可能会迅速导致程序耗尽所有GPU; 好的事，这些情况下的修复通常很简单。这里有几个常见检查事项包括：

一、不要在循环训练中累积历史记录。
默认情况下，涉及需要求导/梯度gradients变量的计算将保存在内存中。计算中避免使用这些变量，例如在跟踪统计数据时，这些变量在循环训练中将超出你内存。相反，您应该分离变量或访问其基础数据。

有时，当可微分变量可能发生时，它可能并不明显。考虑以下循环训练（从源代码删减）：

```
total_loss = 0
for i in range(10000):
    optimizer.zero_grad()
    output = model(input)
    loss = criterion(output)
    loss.backward()
    optimizer.step()
    total_loss += loss
```

在这里，total_loss你的循环训练中积累了历史，因为它loss是一个具有autograd历史的可微变量。你可以通过编写total_loss += float(loss)来解决这个问题。

这个问题的其他例子： 1。

二、释放你不需要的张量和变量。
如果将一个张量或变量分配给本地，Python将不会释放，直到本地超出范围。你可以通过使用del x这样的代码释放。同样，如果将一个张量或变量赋值给对象的成员变量，它将不会释放，直到该对象超出范围。如果你释放了你不需要的变量，内存收益率会提升很多。范围可能比你想象的要大。例如：

```
for i in range(5):
    intermediate = f(input[i])
    result += g(intermediate)
output = h(result)
return output
```

在这里，intermediate 即使在h执行时依然存在，因为它在循环结束后没有释放。你使用完它以后应该使用del intermediate释放它。

三、不要在太大的序列上运行RNN。
通过RNN反向传播所需的内存量与RNN的长度成线性关系; 因此，如果尝试向RNN提供一个时间太长的序列，则会耗尽内存。

这个现象的技术术语是基于时间的反向传播，关于如何实现截断的BPTT有很多参考资料，包括单词language model example; 截断由本论坛帖子中repackage描述的函数处理 。

四、不要使用太大的线性图层。
线性图层 nn.Linear(m, n)使用O(nm)内存：也就是说，权重的内存需求与特征的数量成正比。通过这种方式来超出你的内存是非常容易的（并且记住你至少需要两倍的权重，因为你还需要存储梯度。）

- https://ptorch.com/news/160.html

## SWAP 虛擬內存

Check

```
free -m
```

設立

```
$ sudo fallocate -l 32G /swapfile
$ sudo mkswap /swapfile
$ sudo swapon /swapfile
```

linux 下修改 swap 分区遇到文件忙如何解决？

```
sudo swapoff /swapfile 
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

```

- https://tecadmin.net/how-to-add-swap-memory-on-ubuntu-20-04/

- https://blog.csdn.net/u011046042/article/details/127644869

- https://phoenixnap.com/kb/linux-commands-check-memory-usage

## Pytorch CUDA 顯存不足

```
import torch
torch.cuda.empty_cache()
```

