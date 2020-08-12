# 【小熊猫ML】p1 频率派和贝叶斯派参数估计

小熊猫机器学习终于开课了，这是他第一讲的视频 [【机器学习我到底在学什么】哲学角度聊聊贝叶斯派和频率派，数学角度看看极大似然估计和最大后验估计](https://www.bilibili.com/video/BV1Ea4y1J7Jq)。B站@[软绵绵的小熊猫](https://space.bilibili.com/16241326) 是位数据科学从业者，后续我会跟进这个系列视频，share笔记。常见ML的书会按模型分章介绍，小熊猫这个系列则是尝试介绍将模型连接起来的思想性的东西，但也建议大家手边备本ML的书，如西瓜书、李航统计机器学习。

# 1 以抛硬币的例子简单认识

通过本例简单认识频率派，MLE（maximum likelihood estimation，极大似然估计），贝叶斯派，MAP（Maximum a posteriori estimation，最大后验估计）。

例子很简单：抛10次硬币，9次正面，问正面的概率。

设一个问题的模型参数为 <img src="https://www.zhihu.com/equation?tex=\theta" alt="\theta" class="ee_img tr_noresize" eeimg="1"> ，例子的模型用来预测正面的概率，模型参数就是正面概率。

> 比较绕，不必纠结，在下面MLE的计算中理解。

## 1.1 频率派和MLE

从古典概率发展而来，认为  <img src="https://www.zhihu.com/equation?tex=\theta" alt="\theta" class="ee_img tr_noresize" eeimg="1">  固定，用多次重复试验的统计结果估计  <img src="https://www.zhihu.com/equation?tex=\theta" alt="\theta" class="ee_img tr_noresize" eeimg="1"> ，估计方法为MLE。

认识MLE（maximum likelihood estimation），首先要理解什么是似然函数（Likelihood function）： <img src="https://www.zhihu.com/equation?tex=P(X|\theta)" alt="P(X|\theta)" class="ee_img tr_noresize" eeimg="1"> ， <img src="https://www.zhihu.com/equation?tex=\theta" alt="\theta" class="ee_img tr_noresize" eeimg="1">  未知，输入随机变量  <img src="https://www.zhihu.com/equation?tex=X" alt="X" class="ee_img tr_noresize" eeimg="1"> ，输出 <img src="https://www.zhihu.com/equation?tex=X" alt="X" class="ee_img tr_noresize" eeimg="1"> 对应的概率。

硬币例子的模型是二项分布  <img src="https://www.zhihu.com/equation?tex=B(n,\theta)" alt="B(n,\theta)" class="ee_img tr_noresize" eeimg="1"> ，其中 <img src="https://www.zhihu.com/equation?tex=n" alt="n" class="ee_img tr_noresize" eeimg="1"> 为重复次数， <img src="https://www.zhihu.com/equation?tex=\theta" alt="\theta" class="ee_img tr_noresize" eeimg="1"> 为正面概率。试验10次，9次正面，似然函数  <img src="https://www.zhihu.com/equation?tex=P(X|\theta) = \pmatrix{10\\9} \theta^9 (1-\theta)" alt="P(X|\theta) = \pmatrix{10\\9} \theta^9 (1-\theta)" class="ee_img tr_noresize" eeimg="1"> 。极大似然估计表示为 <img src="https://www.zhihu.com/equation?tex=\theta_{MLE} = \underset{\theta}{\arg\max} P(X|\theta)" alt="\theta_{MLE} = \underset{\theta}{\arg\max} P(X|\theta)" class="ee_img tr_noresize" eeimg="1"> ，意思是求 <img src="https://www.zhihu.com/equation?tex=P(X|\theta)" alt="P(X|\theta)" class="ee_img tr_noresize" eeimg="1"> 极大时参数 <img src="https://www.zhihu.com/equation?tex=\theta" alt="\theta" class="ee_img tr_noresize" eeimg="1"> 的值。因为一般都会假设试验 <img src="https://www.zhihu.com/equation?tex=i.i.d." alt="i.i.d." class="ee_img tr_noresize" eeimg="1"> ，也就是说似然函数会是一个多个试验概率的乘积，故似然函数往往会取对数， <img src="https://www.zhihu.com/equation?tex=\pmatrix{10\\9}" alt="\pmatrix{10\\9}" class="ee_img tr_noresize" eeimg="1"> 不影响求参拿掉， <img src="https://www.zhihu.com/equation?tex=\theta_{MLE} = \underset{\theta}{\arg\max} [ 9\log{\theta} - \log{(1-\theta)]}" alt="\theta_{MLE} = \underset{\theta}{\arg\max} [ 9\log{\theta} - \log{(1-\theta)]}" class="ee_img tr_noresize" eeimg="1"> ，对 <img src="https://www.zhihu.com/equation?tex=\theta" alt="\theta" class="ee_img tr_noresize" eeimg="1">  求导=0 求极值点，可得 <img src="https://www.zhihu.com/equation?tex=\frac{9-10\theta}{\theta(1-\theta)}=0" alt="\frac{9-10\theta}{\theta(1-\theta)}=0" class="ee_img tr_noresize" eeimg="1"> ，故 <img src="https://www.zhihu.com/equation?tex=\theta_{MLE}=0.9" alt="\theta_{MLE}=0.9" class="ee_img tr_noresize" eeimg="1"> ，和统计得出的概率一致。

## 1.2 贝叶斯派和MAP

贝叶斯派认为  <img src="https://www.zhihu.com/equation?tex=\theta" alt="\theta" class="ee_img tr_noresize" eeimg="1">  服从某个分布。根据贝叶斯公式

<img src="https://www.zhihu.com/equation?tex=P(\theta|X) = \frac{P(X|\theta)P(\theta)}{P(X)}
" alt="P(\theta|X) = \frac{P(X|\theta)P(\theta)}{P(X)}
" class="ee_img tr_noresize" eeimg="1">
 <img src="https://www.zhihu.com/equation?tex=P(\theta)" alt="P(\theta)" class="ee_img tr_noresize" eeimg="1">  就是由 <img src="https://www.zhihu.com/equation?tex=\theta" alt="\theta" class="ee_img tr_noresize" eeimg="1">  的分布确定， <img src="https://www.zhihu.com/equation?tex=\theta" alt="\theta" class="ee_img tr_noresize" eeimg="1"> 服从什么分布取决于我们的先验知识，故 <img src="https://www.zhihu.com/equation?tex=P(\theta)" alt="P(\theta)" class="ee_img tr_noresize" eeimg="1"> 称作先验概率。 <img src="https://www.zhihu.com/equation?tex=P(\theta|X)" alt="P(\theta|X)" class="ee_img tr_noresize" eeimg="1">  则是在样本 <img src="https://www.zhihu.com/equation?tex=X" alt="X" class="ee_img tr_noresize" eeimg="1"> 下的条件概率，称作后验概率。MAP表示为 <img src="https://www.zhihu.com/equation?tex=\theta_{MAP} = \underset{\theta}{\arg\max} P(\theta|X)" alt="\theta_{MAP} = \underset{\theta}{\arg\max} P(\theta|X)" class="ee_img tr_noresize" eeimg="1"> ，意思是求后验概率极大时 <img src="https://www.zhihu.com/equation?tex=\theta" alt="\theta" class="ee_img tr_noresize" eeimg="1"> 的值。由于  <img src="https://www.zhihu.com/equation?tex=\theta" alt="\theta" class="ee_img tr_noresize" eeimg="1">  服从某个分布和样本无关， <img src="https://www.zhihu.com/equation?tex=P(X)" alt="P(X)" class="ee_img tr_noresize" eeimg="1"> 可拿掉， <img src="https://www.zhihu.com/equation?tex=\theta_{MAP} = \underset{\theta}{\arg\max}P(X|\theta)P(\theta)" alt="\theta_{MAP} = \underset{\theta}{\arg\max}P(X|\theta)P(\theta)" class="ee_img tr_noresize" eeimg="1"> 。

先验知识不足时，如认为抛硬币正面概率[0,1]内均匀分布， <img src="https://www.zhihu.com/equation?tex=P(\theta)\rarr 0" alt="P(\theta)\rarr 0" class="ee_img tr_noresize" eeimg="1"> ；先验知识丰富，如认为硬币正面概率就是0.5时，那么 <img src="https://www.zhihu.com/equation?tex=P(\theta)" alt="P(\theta)" class="ee_img tr_noresize" eeimg="1"> 只在 <img src="https://www.zhihu.com/equation?tex=\theta=0.5" alt="\theta=0.5" class="ee_img tr_noresize" eeimg="1"> 时=1，其余都为0。无论哪种情况，对后验影响都很大。

需要说明的是MAP不是完整的贝叶斯估计，完整的贝叶斯估计要求出后验 <img src="https://www.zhihu.com/equation?tex=P(\theta|X)" alt="P(\theta|X)" class="ee_img tr_noresize" eeimg="1"> ，这样就能利用 <img src="https://www.zhihu.com/equation?tex=\int_\theta P(x_0|\theta) P(\theta|X)d\theta = \int_\theta P(x_0;\theta|X)d\theta=P(x_0|X)" alt="\int_\theta P(x_0|\theta) P(\theta|X)d\theta = \int_\theta P(x_0;\theta|X)d\theta=P(x_0|X)" class="ee_img tr_noresize" eeimg="1"> ，求取新样本的概率。而根据贝叶斯公式，要求 <img src="https://www.zhihu.com/equation?tex=P(\theta|X)" alt="P(\theta|X)" class="ee_img tr_noresize" eeimg="1"> ，就要先求 <img src="https://www.zhihu.com/equation?tex=P(X)" alt="P(X)" class="ee_img tr_noresize" eeimg="1"> ， <img src="https://www.zhihu.com/equation?tex=P(X)=\int_\theta P(X|\theta)P(\theta)d\theta" alt="P(X)=\int_\theta P(X|\theta)P(\theta)d\theta" class="ee_img tr_noresize" eeimg="1">  是很难积出来的。这点会在后面的视频中讲解。

# 2 以线性回归为例深入理解

线性回归的模型是 <img src="https://www.zhihu.com/equation?tex=Y=W^TX" alt="Y=W^TX" class="ee_img tr_noresize" eeimg="1"> ，最简单的由一些点拟合出一条直线。下面分别用MLE、MAP求参数 <img src="https://www.zhihu.com/equation?tex=W" alt="W" class="ee_img tr_noresize" eeimg="1"> 。

## 2.1 MLE求线性回归参数

考虑噪声 <img src="https://www.zhihu.com/equation?tex=\varepsilon" alt="\varepsilon" class="ee_img tr_noresize" eeimg="1"> ，设 <img src="https://www.zhihu.com/equation?tex=Y=W^TX+\varepsilon" alt="Y=W^TX+\varepsilon" class="ee_img tr_noresize" eeimg="1"> 。根据中心极限定理，设 <img src="https://www.zhihu.com/equation?tex=\varepsilon\sim N(0,\sigma^2)" alt="\varepsilon\sim N(0,\sigma^2)" class="ee_img tr_noresize" eeimg="1"> ，将各种来源的噪声看作不同的随机变量，中心极限定理可简单理解为这些随机变量的累和服从高斯分布。所以， <img src="https://www.zhihu.com/equation?tex=Y\sim N(W^TX,\sigma^2)" alt="Y\sim N(W^TX,\sigma^2)" class="ee_img tr_noresize" eeimg="1"> 。设 <img src="https://www.zhihu.com/equation?tex=x_i,y_i" alt="x_i,y_i" class="ee_img tr_noresize" eeimg="1"> 是样本 <img src="https://www.zhihu.com/equation?tex=i" alt="i" class="ee_img tr_noresize" eeimg="1"> 的值，由前面MLE公式，

<img src="https://www.zhihu.com/equation?tex=\begin{array}{}
\widehat{W}_{MLE} &= \underset{W}{\arg\max}\ \Pi_{i=1}\frac{1}{\sqrt{2\pi}\sigma}exp[-\frac{(y_i-W^Tx_i)^2}{2\sigma^2}]
\\
&= \underset{W}{\arg\max}\ \Sigma_{i=1} [\log{\frac{1}{\sqrt{2\pi}\sigma}} - \frac{(y_i-W^Tx_i)^2}{2\sigma^2}]
\end{array}
" alt="\begin{array}{}
\widehat{W}_{MLE} &= \underset{W}{\arg\max}\ \Pi_{i=1}\frac{1}{\sqrt{2\pi}\sigma}exp[-\frac{(y_i-W^Tx_i)^2}{2\sigma^2}]
\\
&= \underset{W}{\arg\max}\ \Sigma_{i=1} [\log{\frac{1}{\sqrt{2\pi}\sigma}} - \frac{(y_i-W^Tx_i)^2}{2\sigma^2}]
\end{array}
" class="ee_img tr_noresize" eeimg="1">
首先， <img src="https://www.zhihu.com/equation?tex=\log{\frac{1}{\sqrt{2\pi}}}" alt="\log{\frac{1}{\sqrt{2\pi}}}" class="ee_img tr_noresize" eeimg="1"> 不影响计算拿掉，再将剩下部分取反，就变成了 <img src="https://www.zhihu.com/equation?tex=\underset{W}{\arg\min}" alt="\underset{W}{\arg\min}" class="ee_img tr_noresize" eeimg="1"> ，似然函数取极小值时的 <img src="https://www.zhihu.com/equation?tex=W" alt="W" class="ee_img tr_noresize" eeimg="1"> 。由于MLE完全根据样本估计，若样本很小，则结果将波动很大，故再加上一个正则化项，稳定结果，这里采用 L2正则化  <img src="https://www.zhihu.com/equation?tex=\lambda||W||^2" alt="\lambda||W||^2" class="ee_img tr_noresize" eeimg="1"> 。

<img src="https://www.zhihu.com/equation?tex=\begin{array}{}
\widehat{W}_{MLE} &= \underset{W}{\arg\min}\ \Sigma_{i=1} \frac{(y_i-W^Tx_i)^2}{2\sigma^2} + \lambda ||W||^2
\\
&= \underset{W}{\arg\min}\ \Sigma_{i=1} (y_i-W^Tx_i)^2 + \lambda ||W||^2 ,\quad \sigma\text{无关拿掉}
\\
&= \underset{W}{\arg\min}\ L + \lambda ||W||^2,\quad\text{记作L}
\end{array}
" alt="\begin{array}{}
\widehat{W}_{MLE} &= \underset{W}{\arg\min}\ \Sigma_{i=1} \frac{(y_i-W^Tx_i)^2}{2\sigma^2} + \lambda ||W||^2
\\
&= \underset{W}{\arg\min}\ \Sigma_{i=1} (y_i-W^Tx_i)^2 + \lambda ||W||^2 ,\quad \sigma\text{无关拿掉}
\\
&= \underset{W}{\arg\min}\ L + \lambda ||W||^2,\quad\text{记作L}
\end{array}
" class="ee_img tr_noresize" eeimg="1">
这个  <img src="https://www.zhihu.com/equation?tex=L" alt="L" class="ee_img tr_noresize" eeimg="1">  就是线性回归的损失函数，从这也能看出频率派解决问题的步骤：（1）设计模型（2）设计损失函数（3）求解参数。本质上就是采用统计的方法。

## 2.2 MAP求线性回归参数

同MLE，设 <img src="https://www.zhihu.com/equation?tex=Y" alt="Y" class="ee_img tr_noresize" eeimg="1"> 服从正态分布，由MAP公式，

<img src="https://www.zhihu.com/equation?tex=\widehat{W}_{MAP} = \underset{W}{\arg\max}\ P(W|Y)
" alt="\widehat{W}_{MAP} = \underset{W}{\arg\max}\ P(W|Y)
" class="ee_img tr_noresize" eeimg="1">
因为

<img src="https://www.zhihu.com/equation?tex=\begin{array}{}
P(W|Y) &= P(W|y_1,y_2,..)
\\
&= \frac{P(y_1,y_2,..|W)P(W)}{P(Y)}
\\
\underset{W}{\arg\max}\ P(W|Y) &= \underset{W}{\arg\max}\ P(y_1,y_2,..|W)P(W),\quad\text{W和Y无关，拿掉分母}
\\
&= \underset{W}{\arg\max}\ \Pi_{i=1}[P(y_i|W)]P(W),\quad\text{样本 i.i.d.}
\end{array}
" alt="\begin{array}{}
P(W|Y) &= P(W|y_1,y_2,..)
\\
&= \frac{P(y_1,y_2,..|W)P(W)}{P(Y)}
\\
\underset{W}{\arg\max}\ P(W|Y) &= \underset{W}{\arg\max}\ P(y_1,y_2,..|W)P(W),\quad\text{W和Y无关，拿掉分母}
\\
&= \underset{W}{\arg\max}\ \Pi_{i=1}[P(y_i|W)]P(W),\quad\text{样本 i.i.d.}
\end{array}
" class="ee_img tr_noresize" eeimg="1">
正态分布的条件分布仍是正态分布，因为 <img src="https://www.zhihu.com/equation?tex=Y" alt="Y" class="ee_img tr_noresize" eeimg="1"> 服从正态分布，设 <img src="https://www.zhihu.com/equation?tex=P(Y|W)\sim N(0,\sigma^2)" alt="P(Y|W)\sim N(0,\sigma^2)" class="ee_img tr_noresize" eeimg="1"> 。假设 <img src="https://www.zhihu.com/equation?tex=W\sim N(0,\sigma_0^2)" alt="W\sim N(0,\sigma_0^2)" class="ee_img tr_noresize" eeimg="1"> ，强调一下，这里要求的是参数 <img src="https://www.zhihu.com/equation?tex=W" alt="W" class="ee_img tr_noresize" eeimg="1"> ，包括 <img src="https://www.zhihu.com/equation?tex=\sigma_0" alt="\sigma_0" class="ee_img tr_noresize" eeimg="1"> ，则

<img src="https://www.zhihu.com/equation?tex=\begin{array}{}
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
" alt="\begin{array}{}
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
" class="ee_img tr_noresize" eeimg="1">
可以看出，MAP算出的 <img src="https://www.zhihu.com/equation?tex=\widehat{W}_{MAP}" alt="\widehat{W}_{MAP}" class="ee_img tr_noresize" eeimg="1"> 和MLE算出的 <img src="https://www.zhihu.com/equation?tex=\widehat{W}_{MLE}" alt="\widehat{W}_{MLE}" class="ee_img tr_noresize" eeimg="1"> 是一致的，条件是MLE中引入的是L2正则项，并且在MAP中假设先验服从高斯分布。若MLE采用L1正则，MAP先验Laplace分布，推出的结果也是一致的。

# 3 思考

小熊猫最后抛了三个思考题：

Q1 线性回归的MLE中为什么假设误差  <img src="https://www.zhihu.com/equation?tex=\varepsilon" alt="\varepsilon" class="ee_img tr_noresize" eeimg="1">  服从高斯分布？

Q2 为什么L1正则对应Laplace分布的先验，L2正则对应高斯分布的先验？

Q3 对ML的模型进行分类，哪些是频率派，哪些是贝叶斯派？频率派进一步思考损失函数为什么那样设计，贝叶斯派进一步思考如何引入先验分布，为什么引入那样的先验分布？

评论区出现的回答：

A1

![image-20200811215034143](https://raw.githubusercontent.com/pinchenjohnny/Markdown4Zhihu/master/Data/小熊猫机器学习笔记-pinchen/image-20200811215034143.png)

A2

![image-20200811215424148](https://raw.githubusercontent.com/pinchenjohnny/Markdown4Zhihu/master/Data/小熊猫机器学习笔记-pinchen/image-20200811215424148.png)