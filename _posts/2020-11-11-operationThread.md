---
layout: article
title: 线程
key: 100014
tags: 操作系统 线程
category: blog
date: 2020-11-11 00:00:00 +08:00
mermaid: true
---

***fork复制内部，为什么fork返回0？***




# 线程
**定义**
 * 正在运行的一个程序有很多任务，而线程是将任务呈现出细粒度
   * 如何利用CPU处理代码，完成当前进程中的各个子任务。
   * 各个子线程之间共享父进程的代码空间和全局变量，在一个进程中创建多线程代价要小很多


**出现和示例**
 * SUN Solaris操作系统使用的线程叫做UNIX International线程
 * Pthreads线程的头文件是`<pthread.h>`
 * Win32线程是Windows API的一部分。


**特点**
 * 共享地址空间，可以高效共享数据。
 * 多线程的价值在于更好的发挥了多核处理器的效能


**线程共享资源**
 * 堆
 * 全局变量
 * 静态变量
 * 文件


**线程私有资源**
 * 栈
 * 寄存器
 * 程序计数器


**线程的本质**
 * 函数的执行
   * 函数的执行是入口函数
     * CPU从入口函数开始一步一步向下执行，这个过程就叫做线程。
     * 由于函数运行时信息是保存在栈中的，比如返回值，参数，局部变量等等，所以栈是私有的。
     * CPU执行指令的信息会保存在程序计数器。
       * 操作系统可以随时终止线程的运行，所以保存和恢复程序计数器的值就知道线程从哪里暂停的以及从哪里开始运行。

### 线程实现

 * 内核线程实现
   * 操作系统内核支持的线程
     * 通过操纵调度器对线程进行调度，并负责将线程的任务映射到各个处理器上
     * 每个内核线程可以视为内核的一个分身，这种操作系统就有能力同时处理多件事情，支持多线程的内核就叫做多线程内核。
   * 程序一般不会直接去使用内核线程，而是去使用内核线程的接口：轻量级进程(就是我们通常意义上所讲的线程)

 * 用户线程实现
   * 用户线程的建立、同步、销毁和调度完全在用户态中完成
   * 不需要内核的帮助。使用用户线程的优势在于不需要系统内核支援
   * 劣势也在于没有系统内核的支援，所有的线程操作都需要用户程序自己处理。线程的创建、切换和调度都是需要考虑的问题


 * 用户线程加轻量级进程混合实现
   * 将内核线程与用户线程一起使用的实现方式。
   * 在这种混合实现下，既存在用户线程，也存在轻量级进程。

### 线程调度
 * 协同式调度
   * 线程的执行时间由线程本身来控制，线程把自己的工作执行完了之后，要主动通知系统切换到另一个线程上。
   * 好处是实现简单，不会有线程同步问题。
   * 缺点是线程执行时间不可控制，甚至如果一个线程编写有问题，一直不告知系统进行线程切换，那么程序就会一直阻塞在那里。


 * 抢占式调度
   * 每个线程将由系统来分配执行时间，线程的切换不由线程本身来决定。
   * 线程的执行时间是系统可控的，也不会有一个线程导致整个进程阻塞的问题
   * Java使用的线程调度方式就是抢占式调度

内核调度
 * 进程给线程提供了虚拟内存、全局变量等资源。
 * 调度对象：线程
   * 当进程拥有多个线程时，这些线程会共享相同的虚拟内存和全局变量等资源。这些资源在上下文切换时是不需要修改的
   * 线程私有数据（栈、寄存器等）在上下文切换时需要保存

### 线程之间通信方式

**目的**
 * 线程间通信主要目的是为了线程同步
**方法**
 * 锁机制

   * 互斥锁
     * 确保同一时间内只有一个线程能访问共享资源
     * 当资源被占用时，其他试图加锁的线程会进入阻塞状态
     * 当锁释放后，哪个线程能上锁取决于内核调度

   * 读写锁（读模式共享，写模式互斥）
     * 当以写模式加锁的时候，任何其他线程不论以何种方式加锁都会处以阻塞状态
     * 当以读模式加锁时，读状态不阻塞，但是写状态阻塞

   * 自旋锁
     * 上锁受阻时线程不阻塞而是在循环中轮询查看能否获得该锁，没有线程的切换因而没有切换开销，不过对CPU的霸占会导致CPU资源的浪费。

 * posix信号量机制
   * 本质上是一个计数器，可以有PV操作，来控制多个进程或者线程对共享资源的访问
   * 第一组用于进程间通信的，另外一组就是POSIX信号量，信号量原理都是一样的

 * 条件变量
   * 当某个共享数据到达某个值的时候，唤醒等待这个共享数据的线程
   * 要结合互斥锁使用
  ```c
  pthread_mutex_lock(&mutex);
    while(条件为假){
      pthread_cond_wait(&cond, &mutex);
    }
    执行操作
  pthread_mutex_unlock(&mutex);
  ```


### 线程实现中内核线程在linux内核的同步方式
**目的**
 * 同一时间可能有多个内核执行流在执行
 * 因此，内核也需要一些同步机制同步各执行单元对共享数据的访问
   * 在多处理器系统上，更需要一些同步机制来同步不同处理器上的执行单元对共享的数据的访问
**需要同步的条件**
 * 中断
   * 当进程访问某个临界资源发生中断，会进入中断处理程序。中断处理程序也可能访问该临界资源
 * 内核抢占式调度
   * 内核中正在执行的任务被另外一个任务抢占
 * 多处理器并发
   * 每个处理器都可以调度一个进程，多个进程可能会造成并发
**处理方法**
 * 禁用中断
   * 单处理器不可抢占系统来说，系统并发源主要是中断处理
     * 在进行临界资源访问时，进行禁用/使能中断即可以达到消除异步并发源的目的。

 * 原子操作
   * 保证指令在运行时候不会被任何事物或者事件打断，把读和写的行为包含在一步中执行，避免竞争

 * 内存屏障

 * 自旋锁
   * 从一种数据结构中读取数据，写入到另一种数据结构中。
   * 忙等自旋锁只适用于那些在临界区停留很短时间的加锁操作。
     * 线程在等待锁期间会一直占据处理器，如果长时间等待锁会导致处理器效率降低。
     * 而如果线程占用锁只需要短暂的执行一下，那么使用自旋锁更优，因为不需要进行上下文的切换。

 * 信号量
   * 进程无法获取到临界资源的情况下，立即释放处理器的使用权，并睡眠在所访问的临界资源上对应的等待队列上
   * 在临界资源被释放时，再唤醒阻塞在该临界资源上的进程。信号量适用于长时间占用锁的情形。

 * 读-写自旋锁
   * 具体就是在读写场景中了，为读和写提供不同的锁机制。
   * 当一个或者多个读任务时可以并发的持有读锁，但是写锁只能被一个人物所持有的。即对写者共享对读者排斥

 * 读写信号量
   * 和读写自旋锁的思想是一样的

 * mutex体制
   * 是实现互斥的特定睡眠锁
   * 限制
     * 任何时刻只有一个任务可以持有mutex，引用计数只能是1
     * 给mutex上锁必须给其解锁，严格点就是必须在同一上下文中上锁和解锁
     * 持有mutex的进程不能退出

 * 完成变量
   * 内核中一个任务需要发出信号通知另一个任务发生了某个特定事件，利用完成变量使得两个任务实现同步

 * BLK：大内核锁
   * 不用

 * 顺序锁
   * 用于读写共享数据
   * 主要依靠序列计数器，当写数据时，会得到一个锁，序列值会增加。在读取数据前后这两个时间内，序列值都会被读取
   * 如果读取的序列号值相同，表明读操作在进行的过程中没有被写操作打断

 * 关闭内核抢占
   * 自旋锁只能保证共享资源不会被占用，但无法决定CPU线程调度
     * 对于只有一个处理器能够访问到数据，原理上是没有必要加自旋锁的，因为在任何时刻数据的访问者永远只有一位。
     * 但是，如果内核抢占没有关闭，则可能一个新调度的任务就可能访问同一个变量
     * 所以这时候问题是一个任务的访问还没有完成就转到了另一个任务

 * RCU
   * 允许多个读者与写者并发操作而不需要任何锁，这种同步机制可以用于保护通过指针进行访问的数据。比较适合用在读操作很多而写操作极少的情况，可以用来替代读写锁。
     * 内核路由表，路由表的更新是由外部触发的，外部环境的延迟远比内核更新延迟高，在内核更新路由表前实际已经向旧路径转发了很多数据包，RCU读者按照旧路径再多转发几个数据包是完全可以接受的，而且由于RCU的无锁特性，实际上相比有锁的同步机制，内核可以更早生效新的路由表。路由表这个场景，以系统内部短时间的不一致为代价，降低了系统内部与外部世界的不一致时间，同时降低了读者成本。
   * 原因
     *  Linux可以保证指针操作的原子性


### 页面置换算法

**缺页中断**
 一个进程所有地址空间里的页面不必全部常驻内存，在执行一条指令时，如果发现他要访问的页没有在内存中（即存在位为0），那么停止该指令的执行，并产生一个页不存在的异常，对应的故障处理程序可通过从物理内存加载该页的方法来排除故障，之后，原先引起的异常的指令就可以继续执行，而不再产生异常。
**置换算法**
 * 最佳页面置换算法
   * 置换在「未来」最长时间不访问的页面。所以，该算法实现需要计算内存中每个逻辑页面的「下一次」访问时间，然后比较，选择未来最长时间不访问的页面。
   * 实际系统中无法实现，因为程序访问页面时是动态的，我们是无法预知每个页面在「下一次」访问前的等待时间
   * 为了衡量你的算法的效率，你的算法效率越接近该算法的效率，那么说明你的算法是高效的
 * 先进先出置换算法
   * 内存驻留时间很长的页面进行中置换
 * 最近最久未使用的置换算法
   * 选择最长时间没有被访问的页面进行置换，也就是说，该算法假设已经很久没有使用的页面很有可能在未来较长的一段时间内仍然不会被使用。
   * 代价很高
     * 为了完全实现 LRU，需要在内存中维护一个所有页面的链表，最近最多使用的页面在表头，最近最少使用的页面在表尾。
     * 每次访问内存时都必须要更新「整个链表」
 * 时钟页面置换算法
   * 把所有的页面都保存在一个类似钟面的「环形链表」中，一个表针指向最老的页面。
   * 过程
     * 当发生缺页中断时，算法首先检查表针指向的页面
     * 如果它的访问位是 0 就淘汰该页面，并把新的页面插入这个位置，然后把表针前移一个位置
     * 如果访问位是 1 就清除访问位，并把表针前移一个位置，重复这个过程直到找到了一个访问位为 0 的页面为止
 * 最不常用置换算法
   * 选择「访问次数」最少的那个页面，并将其淘汰
   * 代价很高
     * 增加一个计数器来实现，这个硬件成本是比较高的
     * 查找哪个页面访问次数最小，查找链表本身，如果链表长度很大，是非常耗时的，效率不高。
**缺页中断处理**
 * 堆栈中保存程序计数器
 * 保存通用寄存器和其他易失信息
 * 定位到需要的虚拟页面
 * 操作系统检查这个地址是否有效，并检查存取与保护是否一致
   * 如果不一致则杀掉该进程
   * 如果地址有效且没有保护错误发生，系统会检查是否有空闲页框。如果没有空闲页框就执行页面置换算法淘汰一个页面。
 * 如果选择的页框对应的页面发生了修改，即为“脏页面”，需要写回磁盘，并发生一次上下文切换，挂起产生缺页中断的进程，让其他进程运行直至全部把内容写到磁盘。
 * OS会查找要发生置换的页面对应磁盘上的地址，通过磁盘操作将其装入。在装入该页面的时候，产生缺页中断的进程仍然被挂起，运行其他可运行的进程
 * 当发生磁盘中断时表明该页面已经被装入，页表已经更新可以反映其位置，页框也被标记为正常状态。
 * 恢复发生缺页中断指令以前的状态，程序计数器重新指向引起缺页中断的指令
 * 调度引发缺页中断的进程
 * 恢复寄存器和其他状态信息，返回用户空间继续执行


### 线程的同步信号量和锁的互斥量机制

 **区别**
  * 所有权
    * 锁住临界区的锁必须由上锁的线程解开
      * mutex的功能也就限制在了构造临界区
  * 用途
    * 互斥：占有使用权的时候别人不能进入，独占式访问某段程序和内存
    * 同步
      * 一些线程生产一些线程消费，让生产和消费线程保持合理执行顺序。只要是我的信号量满足条件，那么就可以有线程或者进程来使用。
      * 『同步』这个词也可以拆开看，一侧是等待数据的『事件』或者『通知』，一侧是保护数据的『临界区』，所以同步也即同步+互斥。
 **应用**
    * 在做原语开发，do one thing is best
      * linux 内核曾将 semaphore 作为同步原语，后面代码变得较难维护，刷了一把 mutex 变简单了不少还变快了，需要『通知』 的场景则替换为了 completion variable。












## 其他

### 内存对齐

**原因**
 * 加入没有内存对齐机制，数据可以任意存放。浪费开销，访问了两次内存
**规则**
 * 基本类型的对齐值就是sizeof值。如果该成员是c++自带类型如int、char、double等，那么其对齐字节数=该类型在内存中所占的字节数
 * 如果该成员是自定义类型如某个class或者struct，那个它的对齐字节数 = 该类型内最大的成员对齐字节数
 * 编译器可以设置最大对齐值，gcc中默认是#pragma pack(4)。但是类型的实际对齐值与默认对齐值取最小值来
   * 如果定义的字节数为1，就是所有默认字节数直接相加
   * 定义的字节数大于任何一个成员大小时候，不产生任何效果。
   * 如果定义的对齐字节数大于结构体内最小的，小于结构体内最大的话，就按照定义的字节数来计算


### 内存空间的堆和栈

* 栈
  * 栈是由操作系统自动分配的，用于存放函数参数值，局部变量。存储在栈中的数据的生命周期随着函数的执行结束而结束。栈的内存生长方向与堆相反，由高到低，按照变量定义的先后顺序入栈。
* 堆
  * 由用户自己分配的。如果用户不回收，程序结束后由操作系统自动回收。堆的内存地址生长方向与栈相反，由低到高。
**区别**
 * 管理方式
 * 空间大小
 * 分配方式
 * 分配效率
   * 栈：硬件层级对栈提供支持，分配专门的寄存器存放栈的地址，压栈出栈都有专门的指令执行，这就决定了栈的效率比较高。
   * 堆：C/C++提供的库函数或运算符来完成申请与管理，实现机制较为复杂，频繁的内存申请容易产生内存碎片。显然，堆的效率比栈要低得多。
 * 数据结构层面
   * 栈是线性结构
   * 堆是一种特殊的完全二叉树


### 进程虚拟空间布局

**从高到低虚拟空间布局**
 * 内核空间，从C000000-FFFFFFFF
 * 栈区
   * 局部变量
   * 函数调用过程的相关信息，成为栈帧（函数返回地址，一些不适合放在寄存器中的函数参）
   * 暂存算术表达式的计算结果和allocation函数分配的栈内存
 * 内存映射段
   * 内核将硬盘文件的内容直接映射到内存，是一种方便高效文件I/O方式
     * 文件磁盘地址和进程虚拟地址空间中一段虚拟地址的一一对映关系
   * 进程就可以采用指针的方式读写操作这一段内存，而系统会自动回写脏页面到对应的文件磁盘上
     * 对文件的操作而不必再调用read,write等系统调用函数
     * 内核空间对这段区域的修改也直接反映用户空间，从而可以实现不同进程间的文件共享。
   * 原理
     * 由进程启动映射过程，并在虚拟地址空间中为映射创建虚拟映射区域
     * 调用内核空间的系统调用函数mmap（不同于用户空间函数），实现文件物理地址和进程虚拟地址的一一映射关系
     * 进程发起对这片映射空间的访问，引发缺页异常，实现文件内容到物理内存（主存）的拷贝
   * 与常规文件操作的区别
     * 进程发起读文件请求
     * 内核查找文件描述符，定位到内核已打开的文件信息，找到文件的inode
     * 查看文件页是否在缓存中，如果存在则直接返回这片页面
     * 如果不存在，缺页中断，需要定位到该文件的磁盘地址处，将数据从磁盘复制到页缓存中，然后发起页面读写过程，将页缓存中的数据发送给用户（数据传输有两次，如果直接在mmap中只需要一次，只需要建立内存映射段与文件的映射关系）
 * 堆
   * 堆内存是经过字节对齐的空间，以适合原子操作。
   * 堆管理器通过链表管理每个申请的内存，由于堆申请和释放是无序的，最终会产生内存碎片。
   * 堆内存一般由应用程序分配释放，回收的内存可供重新使用。
   * 若程序员不释放，程序结束时操作系统可能会自动回收。
 * BSS
   * 未初始化的全局变量和静态局部变量
   * 初始化值为0的全局变量和静态局部变量
   * 未定义且初值不为0的符号
 * data
   * 已经初始化且初值不为0的全局变量。
   * 数据段属于静态存储区，可读可写
 * 代码段
   * 存放程序执行代码(即CPU执行的机器指令)。一般C语言执行语句都编译成机器代码保存在代码段。通常代码段是可共享的，因此频繁执行的程序只需要在内存中拥有一份拷贝即可
 * 保留区
   * 位于虚拟地址空间的最低部分，未赋予物理地址。
   * 任何对它的引用都是非法的，用于捕捉使用空指针和小整型值指针引用内存的异常情况。
**BSS和data区别**
 * 占用物理文件尺寸
   * BSS段不占用物理文件尺寸，但占用内存空间（不在可执行文件中）。
   * 数据段在可执行文件中，也占用内存空间。
 * 缺页故障的处理
   * 当程序读取data段的数据时候，系统会发生缺页故障，从而分配物理内存。
   * 当程序读取BSS段数据的时候，内核会将其转到一个全零页面，不会发生缺页故障，也不会为期分配物理内存。
 * .exe文件占用
   * bss是不占用.exe文件（可执行文件）空间的，其内容由**操作系统初始化**（清零）
   * 而data却需要占用，其内容由**程序初始化**



### 上下文切换

**上下文**
 * 就是先把前一个任务的 CPU 上下文（也就是 CPU 寄存器和程序计数器）保存起来
 * 加载新任务的上下文到这些寄存器和程序计数器
 * 跳转到程序计数器所指的新位置，运行新任务。
 * 而这些保存下来的上下文，会存储在系统内核中，并在任务重新调度执行时再次加载进来。
 * 这样就能保证任务原来的状态不受影响，让任务看起来还是连续运行。
**CPU切换场景**
 * 进程上下文切换
   * 进程上下文切换的场景
     * 时间片轮转技术下，该进程分配到的时间片耗尽，就会被系统挂起，切换到其他进程
     * 进程在系统资源不足（比如内存不足）时，要等到资源满足后才可以运行，这个时候进程也会被挂起，并由系统调度其他进程运行。
     * 当进程通过睡眠函数 sleep 这样的方法将自己主动挂起时，自然也会重新调度。
     * 当有优先级更高的进程运行时，为了保证高优先级进程的运行，当前进程会被挂起，由高优先级进程来运行
     * 发生硬件中断时，CPU 上的进程会被中断挂起，转而执行内核中的中断服务程序。
   * 从一个进程切换到另一个进程运行
     * 进程的运行包括内核空间和用户空间。
       * 进程既可以在用户空间运行，又可以在内核空间中运行。
       * 进程在用户空间运行时，被称为进程的用户态，
       * 内核空间的时候，被称为进程的内核态。
     * 进程是由内核来管理和调度的，进程的切换只能发生在内核态
   * 切换需要保存资源
     * 用户资源空间（虚拟内存，栈，全局变量等）
     * 内核空间（内核堆栈，寄存器、程序计数器等）。
     * 因此进程在切换的时候，需要把用户态资源和内核态资源保存下来，而加载了下一个进程的内核态后，还需要刷新进程的虚拟内存和用户栈。
 * 线程上下文切换
   * 前后两个线程属于不同进程。此时，因为资源不共享，所以切换过程就跟进程上下文切换是一样。
   * 前后两个线程属于同一个进程。此时，因为虚拟内存是共享的，所以在切换时，虚拟内存这些资源就保持不动，只需要切换线程的私有数据、栈、程序计数器、寄存器等不共享的数据
 * 中断上下文切换
   * 在打断其他进程时，就需要将进程当前的状态保存下来，这样在中断结束后，进程仍然可以从原来的状态恢复运行。
   * 打断了处在用户态的进程，不需要保存和恢复这个进程的虚拟内存、全局变量等用户态资源。、
   * 只需要关注内核资源就行，CPU寄存器，内核堆栈，硬件中断参数等。



### 大端字节、小端字节和转换字节序

**大端字节序**
 高位字节在前，低位字节在后，符合人类读写数值的习惯
**小端字节序**
 低位字节在前，高位字节在后。
**无法统一**
 * 大端字节序：内存的低地址处存放低字节所以在强制转换数据时不需要调整字节的内容（注解：比如把int的4字节强制转换成short的2字节时，就直接把int数据存储的前两个字节给short就行，因为其前两个字节刚好就是最低的两个字节，符合转换逻辑）
   * 大端序更符合人类的习惯，主要用在网络传输和文件存储方面，符号位在所表示的数据的内存的第一个字节中，便于快速判断数据的正负和大小。
 * 小端字节序：CPU做数值运算时从内存中依顺序依次从低位到高位取数据进行运算，直到最后刷新最高位的符号位，这样的运算方式会更高效
 * 各自的优点就是对方的缺点，正因为两者彼此不分伯仲，再加上一些硬件厂商的坚持，因此在多字节存储顺序上始终没有一个统一的标准
**转换字节序**
 * 原因
   * 主要针对主机字节序和网络字节序
     * x86架构的处理器一般都是小端序存储数据
     * 网络字节序是TCP/IP中规定好的数据表示格式（大端）
   * 会使用下列C标准库函数进行字节之间的转换`API arpa/inet.h`
    ```c
    #include <arpa/inet.h>
    uint32_t htonl(uint32_t hostlong);
    uint16_t htons(uint16_t hostshort);
    uint32_t ntohl(uint32_t netlong);
    uint16_t ntohs(uint16_t netshort);
    ```
 * 判断host是否是大端
   ```c
   int i = 1;
   char* p = (char* )&i;
   if(*p == 1){
    // 小端
   }
   else{
    // 大端
   }
   ```



### malloc内存管理
 
 malloc在堆内分配内存
 **堆的结构**
  * 带映射，指针可访问区域
    * 其中由brk指针（是有无映射的两者的区分点）
      * 如果增加堆大小，移动brk指针
      * linux可以使用brk、sbrk系统调用


   ```c
   #include <uistd.h>
   // brk函数将break指针直接设置为某个地址
   int brk(void* addr);
   // sbrk将break指针从当前位置移动increment所指定的增量
   void* sbrk(intptr_t increment);
   ```

  * 不带映射，指针不可访问区域
  * 无法使用空间

 **除了brk、sbrk实现外，还有mmap实现方法**


 **malloc实现方案**
  * brk、sbrk、mmap属于系统调用，每次都要产生系统调用开销（即cpu从用户态切换到内核态的上下文切换，这里要保存用户态数据，等会还要切换回用户态）
  * 申请的内存容易产生碎片
    * 堆是从低地址到高地址，如果低地址的内存没有被释放，高地址的内存就不能被回收
  * malloc采用的是内存池的实现方式
    * 先申请一大块内存，然后将内存分成不同大小的内存块，然后用户申请内存时，直接从内存池中选择一块相近的内存块即可。

   <img src="https://cdn.jsdelivr.net/gh/guaguaupup/cloudimg/data/bins.png" alt="img" style="float: left;" />

    * 内存池保存在bins这个长128的数组中，每个元素都是一双向个链表。
      * malloc将内存分成了大小不同的chunk，然后通过bins来组织起来。
      * malloc将相似大小的chunk（图中可以看出同一链表上的chunk大小差不多）用双向链表链接起来，这样一个链表被称为一个bin。
      * 一共维护了128个bin，并使用一个数组来存储这些bin。
        * 数组中第一个为**unsorted bin**
          * 被用户释放的 chunk 大于 max_fast，或者 fast bins 中的空闲 chunk 合并后，这些 chunk 首先会被放到 unsorted bin 队列中，在进行 malloc 操作的时候，如果在 fast bins 中没有找到合适的 chunk，则malloc 会先在 unsorted bin 中查找合适的空闲 chunk，然后才查找 bins。
          * unsorted bin 可以看做是 bins 的一个缓冲区，增加它只是为了加快分配的速度
        * 数组编号前2到前64的bin为**small bins**，同一个small bin中的chunk具有相同的大小，两个相邻的small bin中的chunk大小相差8bytes。
        * small bins后面的bin被称作**large bins**。
          * large bins中的每一个bin分别包含了一个给定范围内的chunk，其中的chunk按大小序排列。large bin的每个bin相差64字节。
        * 一个**fast bin**。
          * 程序在运行时会经常需要申请和释放一些较小的内存空间。
          * 不大于 max_fast(默认值为 64B)的 chunk 被释放后，首先会被放到 fast bins中，fast bins 中的 chunk 并不改变它的使用标志 P。这样也就无法将它们合并，当需要给用户分配的 chunk 小于或等于 max_fast 时，malloc 首先会在 fast bins 中查找相应的空闲块，然后才会去查找 bins 中的空闲 chunk。
          * 在某个特定的时候，malloc 会遍历 fast bins 中的 chunk，将相邻的空闲 chunk 进行合并，并将合并后的 chunk 加入 unsorted bin 中，然后再将 unsorted bin 里的 chunk 加入 bins 中。
        * 当fast bin和bins都不能满足内存需求时，malloc在top chunk中分配内存
          * 此时brk位于top chunk的顶部，移动brk指针，即可扩充top chunk的大小
          * top chunk大小超过128k(可配置)时，会触发malloc_trim操作，调用sbrk(-size)将内存归还操作系统。
        * 当fast bin、bins和top chunk不能满足，malloc会从mmap来直接使用内存映射来将页映射到进程空间，这样的chunk释放时，直接解除映射，归还给操作系统。（极限大的时候）
        * Last remainder是另外一种特殊的chunk，就像top chunk和mmaped chunk一样，不会在任何bins中找到这种chunk。
          * 当需要分配一个small chunk,但在small bins中找不到合适的chunk，如果last remainder chunk的大小大于所需要的small chunk大小，last remainder chunk被分裂成两个chunk，
          * 其中一个chunk返回给用户，另一个chunk变成新的last remainder chunk。（这个应该是fast bins中也找不到合适的时候，用于极限小的）

 **内存分配流程**
  * 分配内存<512字节，则通过内存大小定位到smallbins对应的index上(floor(size/8))
    * 如果smallbins[index]为空，进入步骤3
    * 如果smallbins[index]非空，直接返回第一个chunk
  * 如果分配内存>512字节，则定位到largebins对应的index上
    * 如果largebins[index]为空，进入步骤3
    * 如果largebins[index]非空，扫描链表，找到第一个大小最合适的chunk，如size=12.5K，则使用chunk B，剩下的0.5k放入unsorted_list中
  * 遍历unsorted_list，查找合适size的chunk，如果找到则返回；否则，将这些chunk都归类放到smallbins和largebins里面
  * 如果没有找到chunk，index++从更大的链表中查找，直到找到合适大小的chunk为止，找到后将chunk拆分，并将剩余的加入到unsorted_list中
  * 如果还没有找到，那么使用top chunk
  * 或者，内存<128k，使用brk；内存>128k，使用mmap获取新内存

 **调用free**
  * 将用户释放的内存块连接到空闲链上，空闲链会被切成很多的小内存片段
    * 如果这时用户申请一个大的内存片段，那么空闲链上可能没有可以满足用户要求的片段了。malloc函数请求延时
    * 在空闲链上翻箱倒柜地检查各内存片段，对它们进行整理，将相邻的小空闲块合并成较大的内存块。

 **free内存碎片**
  * free的情况
    * chunk和top chunk相邻
      * 和top chunk合并
        * 操作系统需要brk指针位置以上才认为是放入系统内存空间中，因此对于顶部的内存资源，直接通过移动brk指针完成
    * chunk和top chunk不相邻
      * 出现内存碎片，brk指针没有变化，但有chunk的内存被释放
      * 将内存空间交给unsorted_list



### 写时拷贝原理
 **目的**
  * fork系统调用子线程的时候为了保证速度不会将父进程的所有内存copy，与父进程共用相同的内存页
  * 如果有多个调用者同时请求相同资源（如内存或磁盘上的数据存储），他们会共同获取相同的指针指向相同的资源
  * 某个调用者试图修改资源的内容时，系统才会真正复制一份专用副本给该调用者，而其他调用者所见到的最初的资源仍然保持不变
 **原理**
  * fork()之后，kernel把父进程中所有的内存页的权限都设为read-only，然后子进程的地址空间指向父进程。当父子进程都只读内存时，相安无事。
  * 当其中某个进程写内存时，CPU硬件检测到内存页是read-only的，于是触发页异常中断（page-fault），陷入kernel的一个中断例程。
  * 中断例程中，kernel就会把触发的异常的页复制一份，于是父子进程各自持有独立的一份。这样父进程和子进程都有了属于自己独立的页。
  * 子进程可以执行exec()来做自己想要的功能。



### cache

 CPU和主存储器DRAM之间的一块高速缓冲存储器

 **工作**
  * CPU在访问存储器的时候，同时把地址发送给MMU中的TLB以及Cache
  * CPU会在TLB中查找最终的RPN（Real Page Number），也就是真实的物理页面，如果找到了，就会返回相应的物理地址。同时，CPU通过cache编码地址中的Index，也可以很快找到相应的Cache line组
  * 如果TLB命中后，会返回一个真实的物理地址，将cache line中存放的地址和这个转换出来的物理地址进行比较，如果相同并且状态位匹配，那么就会发生cache命中。
    * 如果cache miss，那么CPU就需要重新从存储器中获取数据，然后再将其存放在cache line中。


 **为了进一步提升性能，引入多级cache**

### memory barrier

 **目的**
  * 为了提高性能而采取乱序执行，这可能会导致程序运行不符合我们预期
  * 内存屏障就是一类同步屏障指令，是CPU或者编译器在对内存随机访问的操作中的一个同步点
    * 只有在此点之前的所有读写操作都执行后才可以执行此点之后的操作。


<!-- ### fork复制内部，为什么fork返回0？ -->

带锁是对数据上锁
带上锁，如果可以抢占则可能会死锁
带上锁，如何可以中断，则可能死锁


[锁、中断和抢占的关系](http://blog.guorongfei.com/2014/09/06/linux-interrupt-preemptive-lock/)



### 锁
 **目的**
  * 保证临界区代码的安全

 **单核**
  * 关闭 CPU 中断，使其不能暂停当前请求而处理其他请求，从而达到赋值“锁”对应的内存空间的目的。
 **多核**
  * 锁总线和缓存一致性技术（详情看这里），可以实现在单一时刻，只有某个CPU里面某一个核能够赋值“锁”对应的内存空间，从而达到锁的目的
    * 原因
      * CPU 有自己的内部缓存，根据一些规则将内存中的数据读取到内部缓存中来，以加快频繁读取的速度。那么这样就会造成cpu寄存器中的值和内存中的值出现不匹配的现象。
    * 总线锁定
      * CPU1 要做 i++ 操作的时候，其在总线上发出一个 LOCK 信号，其他处理器就不能操作缓存了该变量内存地址的缓存，也就是阻塞了其他CPU，使该处理器可以独享此共享内存。
    * 缓存一致性
      * 当某块 CPU 对缓存中的数据进行操作了之后，就通知其他 CPU 放弃储存在它们内部的缓存，或者从主内存中重新读取
 **开销**
  * 处理器提供的原子操作指令CAS，处理器会用轮询的方式试图获得锁，在处理器（包括多核）架构里这是必不可少的机制
  * 内核提供的锁系统调用，在被锁住的时候会把当前线程置于睡眠（阻塞）状态
    * 实际上我们在编程的时候并不会直接调用这两种机制，而是使用编程语言所带函数库里的锁方法，锁方法内部混合使用这两种机制。
      * pthread_mutex
      * 一把锁本质上只是一个int类型的变量，占用4个字节内存并且内存边界按4字节对齐。
      * 加锁的时候先用trylock方法（内部使用的是CAS指令）来尝试获得锁
      * 如果无法获得锁，则调用系统调用sys_futex来试图获得锁
      * 这时候如果还不能获得锁，当前线程就会被阻塞。
        * 如果锁不存在冲突，每次获得锁和释放锁的处理器开销仅仅是CAS指令的开销，在x86-64处理器上，这个开销只比一次内存访问（无cache）高一点（大概是1.3倍），14ns左右
        * 锁冲突的情况：运行的结果是双核机器上消耗大约3400ns
  * 线程上下文切换开销、调度器开销（把线程从睡眠改成就绪或者把就运行态改成阻塞）、后续上下文切换cache miss、跨处理器调度开销
 **锁优化**
  * 锁冲突的次数
    * 使用更细粒度的锁，可以减少锁冲突
      * 空间粒度
        * 哈希表包含一系列哈希桶，为每个桶设置一把锁，空间粒度就会小很多－－哈希值相互不冲突的访问不会导致锁冲突，这比为整个哈希表维护一把锁的冲突机率低很多
      * 时间粒度
        * 只包含必要的代码段，尽量缩短获得锁到释放锁之间的时间
        * 最重要的是，绝不要在锁中进行任何可能会阻塞的操作。
        * 使用读写锁也是一个很好的减少冲突的方式，读操作之间不互斥，大大减少了冲突。
    * 读写锁例子
      * 单向链表中的插入/删除操作很少，主要操作是搜索，
        * 基于单一锁的方法性能会很差。
        * 考虑使用读写锁，即 pthread_rwlock_t，允许多个线程同时搜索链表，插入和删除操作仍然会锁住整个链表
      * 执行的插入和搜索操作数量差不多相同，但是删除操作很少
        * 在插入期间锁住整个链表是不合适的
        * 允许在链表中的分离点（disjoint point）上执行并发插入，同样使用基于读写锁的方式。在两个级别上执行锁定
        * 链表有一个读写锁，各个节点包含一个互斥锁，在插入期间，写线程在链表上建立读锁，然后继续处理。
          * 在插入数据之前，锁住要在其后添加新数据的节点，插入之后释放此节点，然后释放读写锁。
          * 删除操作在链表上建立写锁。不需要获得与节点相关的锁；互斥锁只建立在某一个操作节点之上，大大减少锁冲突的次数。
    * 行为优化
      * sys_futex消耗很高，一个值得考虑的优化方式是先循环调用 CAS 来尝试获得锁，在若干次失败后再进入内核真正加锁。
        * 只能在多处理器的系统里起作用（得有另一个处理器来解锁，否则自旋锁无意义）。在glibc的pthread实现里，通过对pthread_mutex设置PTHREAD_MUTEX_ADAPTIVE_NP属性就可以使用这个机制。
      * 读多写一的情况用double buffer

 **futex**
  * futex是linux pthread mutex的实现
  * 提出背景
    * Unix系统中，IPC，进程间同步机制都是对一个内核对象操作来完成的，其提供了共享的状态信息和原子操作，用来管理互斥锁并且通知阻塞的进程
      * 某个进程进入互斥区，到再从某个互斥区出来这段时间，常常是没有进程也要进这个互斥区或者请求同一同步变量的。
      * 这个进程也要陷入内核去看看有没有人和它竞争，退出的时侯还要陷入内核去看看有没有进程等待在同一同步变量上，有的话需要唤醒等待的进程。这些不必要的系统调用(或者说内核陷入)造成了大量的性能开销
  * 原理
    * 用户态和内核态混合的同步机制
    * 同步的进程间通过 mmap 共享一段内存，futex 变量就位于这段共享的内存中且操作是原子的
      * 当进程尝试进入互斥区或者退出互斥区的时候，先去查看共享内存中的 futex 变量，如果没有竞争发生，就不用再执行系统调用了。当通过访问 futex 变量后进程发现有竞争发生，则还是得执行系统调用去完成相应的处理（wait 或者 wake up）。简单的说，futex 就是通过在用户态的检查，（motivation）如果了解到没有竞争就不用陷入内核了，大大提高了 low-contention 时候的效率。

  * linux mutex 
    * 用的内存共享变量来实现的，如果共享变量建立在进程内，它就是一个线程锁，如果它建立在进程间共享内存上，那么它是一个进程锁。pthread_mutex_t 中的 `_lock` 字段用于标记占用情况，先使用CAS判断`_lock`是否占用，若未占用，直接返回。否则，通过`__lll_lock_wait_private` 调用`SYS_futex `系统调用迫使线程进入沉睡。 CAS是用户态的 CPU 指令，若无竞争，简单修改锁状态即返回，非常高效，只有发现竞争，才通过系统调用陷入内核态。所以，FUTEX是一种用户态和内核态混合的同步机制，它保证了低竞争情况下的锁获取效率。
  

