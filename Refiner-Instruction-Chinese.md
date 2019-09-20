# 标注修改器使用手册 V1.0

## 开始

1. 打开` ./src/utils/refine_openpose.py`，修改`dir`参数为本地数据集地址。
2. 将`annotation_all.txt`文件放入该目录(与各分组同目录)下，注释其他函数并运行`distribute(dir)。`
3. 运行完成后，注释`distribute(dir)`，将`dir`修改为数据集下某一分组目录（标注完一组后也依此修改分组即可标注下一组）。
4. 运行`refine()`，其中`os`参数按照运行系统填入`mac or win`（程序开始运行时可能没有出现弹窗，请留意任务栏）。
5. 笔记本建议插电使用。

## 修改关节坐标

1. 鼠标左键点击目标关节即可绿色高亮显示，拖动便可修改其坐标。

## 删除标定人物

1. 鼠标右键点击目标人物任意关节，即会红色高亮显示人物所有关节，按下`Backspace`键即可完成删除。

## 添加关节

1. 目前仅支持单人体关节添加，因此请确保添加关节前已将多余人物删除。
2. 按下空格键即可进入添加模式，请在控制台的指示下输入需要添加的关节号（关节表见下表），并回车确认。
3. 点击需要添加关节所在的位置，即可完成添加。

|     Joint      | Number |
| :------------: | :----: |
|      Nose      |   0    |
|      Neck      |   1    |
| Right Shoulder |   2    |
|  Right Elbow   |   3    |
|  Right Wrist   |   4    |
| Left Shoulder  |   5    |
|   Left Elbow   |   6    |
|   Left Wrist   |   7    |
|   Right Hip    |   8    |
|   Right Knee   |   9    |
|  Right Ankle   |   10   |
|    Left Hip    |   11   |
|   Left Knee    |   12   |
|   Left Ankle   |   13   |
|   Right Eye    |   14   |
|    Left Eye    |   15   |
|   Right Ear    |   16   |
|    Left Ear    |   17   |

## 保存修改

1. 按下`Enter`将切换到下一张图像，任何所做的修改当且仅当按下`Enter`并切换到下一张时才被保存（在删除模式下无法进行切换，请先在任一处点击鼠标退出删除状态）。

## 恢复初始状态

1. 删除分组目录下的`refined.txt`即可清除所有修改。