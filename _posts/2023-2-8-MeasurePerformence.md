

# Measure Performence




#### speedup

 * 顺序执行时间 / 并行执行时间

 * 在顺序执行的执行时间，选择测试中的最好的measure，使用找到的baseline，而不是自己写的baseline



#### Efficency

 * 用于加速的处理器个数P / P的总数 

 **用处**
  * 常用于衡量多cpu core的程序
  * 希望是接近1，大多数情况都是小于1的
  * 超线程加速：使用额外资源，例如cache等，所以efficency大于1
  * 对于GPU来说，efficency衡量的就不是很有效了。
    * 对于GPU，一般衡量efficency是将使用的资源与峰值进行比较。



#### Scalability

 * 衡量当计算资源增加时，我们的硬件和软件利用计算资源的能力。表现为当processors增加的时候，speed up的值是多少，好的表现为近似线性关系




 * 下图的这个例子里，就不是scalable code

 ![](https://p6-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/aecc232ed65d43ca85ec6d03822c3067~tplv-k3u1fbpfcp-zoom-in-crop-mark:4536:0:0:0.awebp?)



#### 阿姆达尔定律

 * $speedup=1 / ( s + p / N )$
   * 当问题规模一定时，提高硬件资源N，以提高speedup。

#### 古斯塔夫森定律
 * $W(s)=(1-p)*W +spW$
  
  $W$：在系统加入新资源前，$T$时间内可以完成的任务
  $p$：系统中，可以使用并行计算的任务的比例
  $(1-p)$：系统中，不可以使用并行计算的任务的比例
  $s$：计算资源（初始系统中的计算资源为1）
  $W(s)$：$T$时间内可以完成的任务，是相关的$s$函数

  当问题规模增大的时候，对于相同问题，通过增加硬件资源数可以让work/processor保持稳定。
