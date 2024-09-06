# NEU-Seg
Hnet ---> 模仿unet的架构，但是加入注意力模块
Upsampler --> 上采样模块
Downsampler --> 下采样模块
注意力模块思路：把图片reshape成[batchsize,channel,-1]然后使用点积注意力或者其他的注意力方式。在完成注意力后reshape成原来的形状，裁剪然后concat
