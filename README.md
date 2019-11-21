# Pytorch implementation of SiamFC

####  此项目主要是针对作者对SiamFC的Pytorch实现做了更多的中文注释

原始[readme.txt](https://github.com/fengchengAI/SiamFC-PyTorch/blob/master/README1.md)修改为readme１.txt，建议在看代码前看此readme，在测试程序时看readme１.txt

[SiamFC: Fully-Convolutional Siamese Networksfor Object Tracking](https://arxiv.org/pdf/1606.09549.pdf)

是目标跟踪的开山之作，主要是设计了两个孪生网络，计算之间的相似度，如下所示：

![siamfc](https://raw.githubusercontent.com/fengchengAI/Pictures/master/siamfc.png)

z为样本图片，即example，尺寸为127，在经过基础网络后变成6	

x为待搜索图片，即instance，尺寸为255，在经过基础网络后变成22

然后将6作为卷积核在22上卷积，得到17的response的特征图，也即是相识度的映射．



在文章中有一个比较重要的概念`context`
![siamfc-context](https://raw.githubusercontent.com/fengchengAI/Pictures/master/siamfc-context.png)
)
在论文中有2.4节的公式7中有如下公式;  

$$s(w+2p)\times s(h+2p) = A  \tag{7}$$

$$A = 127^2;	p = (w+h)/4$$

则：$s= \sqrt{\frac{A}{(w+2p)\times(h+2p)}}$

该公式是在example上的一个应用，	公式７可以近似看成是一个卷积核为１，填充为p，步长为ｓ的卷积操作，然后得到feature_map为Ａ

如上图所示：

其中紫色为原始gt_bound, 根据$p = (w+h)/4$得到$p=24$，然后对Ｗ,H进行填充，会得到112,80的大小，则$s = \frac{127}{\sqrt{112*80}} = 1.34$,s为缩放因子，红色框为真实框的缩放后的结果

在真实操作中，以上的物理含义是：在图片中根据gt_bound（紫色）的w,h会在原始gt_bound中心周围进行填充，得到一个相对于gt_bound较大的包含填充的框（橙色），然后会以这个框，在原始图片上进行裁剪，再resize到相应的尺寸{127,255}

### 备注

在本项目中，为了简单，并没有对w,h填充相同的大小，而是根据$\sqrt{(w+2p)\times(h+2p)}$作为边长得到一会正方形的包含填充的框（橙色），再resize

`论文中主要细节如下：`

训练所用到数据全部进行了基于gt_bound中心的裁剪，即如果存在一个img,及其对应box，

$s\_z = \sqrt{(w+2p)\times(h+2p)}	$     ，    $scale\_z = \frac{127}{s\_z}$    

$scale\_x = scale\_z$  ，  $s\_x = \frac{255}{s\_x}$　　　

s为图中所示的橙色框，scale为缩放比，z表示example，x表示instance（上面的标识为代码中的标识）

如果该图片作为example，则计算$s\_z$将图像裁剪至以box的center_x，和center_y为中心，边长为$s\_z$的矩形，然后resize到127，对于instance图像也是一样的

上面对于训练阶段对图像的裁剪处理，作者在线下进行完成，并将图片打包为tqdm以方便高速读取．

在损失阶段，由于instance和example都进行了中心化，则response_map的中心肯定是得分最高的．所以对于真实label的构建，可以用一个类似与圆的２维矩阵表示，最中间最大，越发散，越小，当超过一定的半径时认为为负值．

损失函数可以对response_map的每个元素，进行最简单的`logistic loss`回归

在测试阶段：

首先会根据第一张图片，初试化网络，与训练阶段类似，第一帧会有一个box，然后将box的信息[center_x,center_y,w,h]，然后第n+1帧会根据上一帧的box信息继续将instance进行裁剪，此时由于目标的位置可能已经发生改变，所以最大的response_map的值可能不是最中心的位置，将最大值距离中心的偏差映射到原始图片中就是下一帧box的center_x和center_y信息．关于w,h的回归，在训练阶段定义了３个scale即是size的缩放，然后在下一帧会有增大或者减少，或者不变．

### 缺点

在测试阶段会发现较大的缺点，就说有关box的w,h刻度变化，不够灵活，即

(w_new, h_new) = s * (w_old, h_old)　即w和h是相同的缩放比



推荐一个不错的Pytorch实现，其标注比较详细完善，只是没有测试阶段的代码

[Pytorch-SiamFC](https://github.com/rafellerc/Pytorch-SiamFC)
