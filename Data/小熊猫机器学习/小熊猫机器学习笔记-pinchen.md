# 【小熊猫ML】p1 频率派和贝叶斯派参数估计

小熊猫机器学习终于开课了，这是他第一讲的视频 [【机器学习我到底在学什么】哲学角度聊聊贝叶斯派和频率派，数学角度看看极大似然估计和最大后验估计](https://www.bilibili.com/video/BV1Ea4y1J7Jq)。B站@[软绵绵的小熊猫](https://space.bilibili.com/16241326) 是位数据科学从业者，后续我会跟进这个系列视频，share笔记。常见ML的书会按模型分章介绍，小熊猫这个系列则是尝试介绍将模型连接起来的思想性的东西，但也建议大家手边备本ML的书，如西瓜书、李航统计机器学习。

# 1 以抛硬币的例子简单认识

通过本例简单认识频率派，MLE（maximum likelihood estimation，极大似然估计），贝叶斯派，MAP（Maximum a posteriori estimation，最大后验估计）。

例子很简单：抛10次硬币，9次正面，问正面的概率。

设一个问题的模型参数为$\theta$，例子的模型用来预测正面的概率，模型参数就是正面概率。

> 比较绕，不必纠结，在下面MLE的计算中理解。

## 1.1 频率派和MLE

从古典概率发展而来，认为 $\theta$ 固定，用多次重复试验的统计结果估计 $\theta$，估计方法为MLE。

认识MLE（maximum likelihood estimation），首先要理解什么是似然函数（Likelihood function）：$P(X|\theta)$，$\theta$ 未知，输入随机变量 $X$，输出$X$对应的概率。

硬币例子的模型是二项分布 $B(n,\theta)$，其中$n$为重复次数，$\theta$为正面概率。试验10次，9次正面，似然函数 $P(X|\theta) = \pmatrix{10\\9} \theta^9 (1-\theta)$。极大似然估计表示为$\theta_{MLE} = \underset{\theta}{\arg\max} P(X|\theta)$，意思是求$P(X|\theta)$极大时参数$\theta$的值。因为一般都会假设试验$i.i.d.$，也就是说似然函数会是一个多个试验概率的乘积，故似然函数往往会取对数，$\pmatrix{10\\9}$不影响求参拿掉，$\theta_{MLE} = \underset{\theta}{\arg\max} [ 9\log{\theta} - \log{(1-\theta)]}$，对$\theta$ 求导=0 求极值点，可得$\frac{9-10\theta}{\theta(1-\theta)}=0$，故$\theta_{MLE}=0.9$，和统计得出的概率一致。

## 1.2 贝叶斯派和MAP

贝叶斯派认为 $\theta$ 服从某个分布。根据贝叶斯公式
$$
P(\theta|X) = \frac{P(X|\theta)P(\theta)}{P(X)}
$$
$P(\theta)$ 就是由$\theta$ 的分布确定，$\theta$服从什么分布取决于我们的先验知识，故$P(\theta)$称作先验概率。$P(\theta|X)$ 则是在样本$X$下的条件概率，称作后验概率。MAP表示为$\theta_{MAP} = \underset{\theta}{\arg\max} P(\theta|X)$，意思是求后验概率极大时$\theta$的值。由于 $\theta$ 服从某个分布和样本无关，$P(X)$可拿掉，$\theta_{MAP} = \underset{\theta}{\arg\max}P(X|\theta)P(\theta)$。

先验知识不足时，如认为抛硬币正面概率[0,1]内均匀分布，$P(\theta)\rarr 0$；先验知识丰富，如认为硬币正面概率就是0.5时，那么$P(\theta)$只在$\theta=0.5$时=1，其余都为0。无论哪种情况，对后验影响都很大。

需要说明的是MAP不是完整的贝叶斯估计，完整的贝叶斯估计要求出后验$P(\theta|X)$，这样就能利用$\int_\theta P(x_0|\theta) P(\theta|X)d\theta = \int_\theta P(x_0;\theta|X)d\theta=P(x_0|X)$，求取新样本的概率。而根据贝叶斯公式，要求$P(\theta|X)$，就要先求$P(X)$，$P(X)=\int_\theta P(X|\theta)P(\theta)d\theta$ 是很难积出来的。这点会在后面的视频中讲解。

# 2 以线性回归为例深入理解

线性回归的模型是$Y=W^TX$，最简单的由一些点拟合出一条直线。下面分别用MLE、MAP求参数$W$。

## 2.1 MLE求线性回归参数

考虑噪声$\varepsilon$，设$Y=W^TX+\varepsilon$。根据中心极限定理，设$\varepsilon\sim N(0,\sigma^2)$，将各种来源的噪声看作不同的随机变量，中心极限定理可简单理解为这些随机变量的累和服从高斯分布。所以，$Y\sim N(W^TX,\sigma^2)$。设$x_i,y_i$是样本$i$的值，由前面MLE公式，
$$
\begin{array}{}
\widehat{W}_{MLE} &= \underset{W}{\arg\max}\ \Pi_{i=1}\frac{1}{\sqrt{2\pi}\sigma}exp[-\frac{(y_i-W^Tx_i)^2}{2\sigma^2}]
\\
&= \underset{W}{\arg\max}\ \Sigma_{i=1} [\log{\frac{1}{\sqrt{2\pi}\sigma}} - \frac{(y_i-W^Tx_i)^2}{2\sigma^2}]
\end{array}
$$
首先，$\log{\frac{1}{\sqrt{2\pi}}}$不影响计算拿掉，再将剩下部分取反，就变成了$\underset{W}{\arg\min}$，似然函数取极小值时的$W$。由于MLE完全根据样本估计，若样本很小，则结果将波动很大，故再加上一个正则化项，稳定结果，这里采用 L2正则化 $\lambda||W||^2$。
$$
\begin{array}{}
\widehat{W}_{MLE} &= \underset{W}{\arg\min}\ \Sigma_{i=1} \frac{(y_i-W^Tx_i)^2}{2\sigma^2} + \lambda ||W||^2
\\
&= \underset{W}{\arg\min}\ \Sigma_{i=1} (y_i-W^Tx_i)^2 + \lambda ||W||^2 ,\quad \sigma\text{无关拿掉}
\\
&= \underset{W}{\arg\min}\ L + \lambda ||W||^2,\quad\text{记作L}
\end{array}
$$
这个 $L$ 就是线性回归的损失函数，从这也能看出频率派解决问题的步骤：（1）设计模型（2）设计损失函数（3）求解参数。本质上就是采用统计的方法。

## 2.2 MAP求线性回归参数

同MLE，设$Y$服从正态分布，由MAP公式，
$$
\widehat{W}_{MAP} = \underset{W}{\arg\max}\ P(W|Y)
$$
因为
$$
\begin{array}{}
P(W|Y) &= P(W|y_1,y_2,..)
\\
&= \frac{P(y_1,y_2,..|W)P(W)}{P(Y)}
\\
\underset{W}{\arg\max}\ P(W|Y) &= \underset{W}{\arg\max}\ P(y_1,y_2,..|W)P(W),\quad\text{W和Y无关，拿掉分母}
\\
&= \underset{W}{\arg\max}\ \Pi_{i=1}[P(y_i|W)]P(W),\quad\text{样本 i.i.d.}
\end{array}
$$
正态分布的条件分布仍是正态分布，因为$Y$服从正态分布，设$P(Y|W)\sim N(0,\sigma^2)$。假设$W\sim N(0,\sigma_0^2)$，强调一下，这里要求的是参数$W$，包括$\sigma_0$，则
$$
\begin{array}{}
\underset{W}{\arg\max}\ P(W|Y)
&= \underset{W}{\arg\max}\ \Pi_{i=1}[\frac{1}{\sqrt{2\pi}\sigma}exp[-\frac{(y_i-W^Tx_i)^2}{2\sigma^2}]]\ \frac{1}{\sqrt{2\pi}\sigma_0}exp[-\frac{W^2}{2\sigma_0^2}]
\\
&= \underset{W}{\arg\max}\ \Sigma_{i=1}[\log{\frac{1}{\sqrt{2\pi}\sigma}} - \frac{(y_i-W^Tx_i)^2}{2\sigma^2}]\ + [\log{\frac{1}{\sqrt{2\pi}\sigma_0}} - \frac{W^2}{2\sigma_0^2}]
\\
&= \underset{W}{\arg\min}\ \Sigma_{i=1}\frac{(y_i-W^Tx_i)^2}{2\sigma^2} + \frac{W^2}{2\sigma_0^2},\quad\text{去掉不影响的项，并取反变argmin}
\\
&= \underset{W}{\arg\min}\ \Sigma_{i=1}(y_i-W^Tx_i)^2 + (\frac{\sigma}{\sigma_0}W)^2,\quad\text{两边同时乘以}2\sigma^2
\\
\widehat{W}_{MAP} &= \underset{W}{\arg\min}\ \Sigma_{i=1}(y_i-W^Tx_i)^2 + \lambda||W||^2,\quad\text{记作}\lambda
\end{array}
$$
可以看出，MAP算出的$\widehat{W}_{MAP}$和MLE算出的$\widehat{W}_{MLE}$是一致的，条件是MLE中引入的是L2正则项，并且在MAP中假设先验服从高斯分布。若MLE采用L1正则，MAP先验Laplace分布，推出的结果也是一致的。

# 3 思考

小熊猫最后抛了三个思考题：

Q1 线性回归的MLE中为什么假设误差 $\varepsilon$ 服从高斯分布？

Q2 为什么L1正则对应Laplace分布的先验，L2正则对应高斯分布的先验？

Q3 对ML的模型进行分类，哪些是频率派，哪些是贝叶斯派？频率派进一步思考损失函数为什么那样设计，贝叶斯派进一步思考如何引入先验分布，为什么引入那样的先验分布？

评论区出现的回答：

A1

![image-20200811215034143](小熊猫机器学习笔记-pinchen/image-20200811215034143.png)

A2

![image-20200811215424148](小熊猫机器学习笔记-pinchen/image-20200811215424148.png)