# deeplab-paddle
paddle version for deeplab

1. 基于paddleSeg
2. 解压模型后修改run_eval.sh 中的模型路径即可得到最终结果。
3. crf很慢，因此默认是关闭状态，若要开启请在val.py 中输入参数--use_crf即可激活
4. 复现精度为75.1
