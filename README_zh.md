# DramaSR-LRM: 

## 环境建立

确保你拥有能运行LLaMA-Factory和verl-tool框架的包环境即可。

## 数据准备

我们提供了10部中文剧+3部英文剧的剧集数据，位于`LRM/<剧集名>`文件夹下。
你可以通过本节指引的方法使用api生成推理数据，并从中划分出用于sft和rl训练的数据。

### 推理数据生成

此步骤可以实现使用api生成推理数据。
进入`DramaSR-LRM/LRM`目录，运行`tool_master.py`，并指定`--series_name`为剧集名称，`--end_name`为文件输出名后缀。你需要在`chat.py`中填写你的api_key和base_url。
api生成的推理数据会储存在`test_result`文件夹中。

### 数据分配

此步骤可以实现按照配比从推理数据中提取sft数据和rl数据。
进入`DramaSR-LRM/training`目录，运行`data_split.py`，并指定`--series_name`，`--end_name`参数与上一步Data Generation时相同。程序会在`DramaSR-LRM/LRM/test_result`中加载所需的数据。其余需要指定的参数含义于`data_split.py`文件中说明。你可以自定义自己的data_split文件，以调节配比方案。
此步骤生成的数据分配结果会储存在当前`training`目录，包含split_data和meta_data。

### SFT数据生成

此步骤可以实现依据数据分配结果，将其中的sft数据处理成训练所需格式。
进入`DramaSR-LRM/training`目录，运行`generate_sft_data.py`，并指定`--input_filename`为Data Split生成的split_data文件，`--dataset_format`为`alpaca`（暂不支持其他格式）。
此步骤生成的sft数据会储存在当前`training`目录。

**项目在`training`目录中提供了生成好的SFT数据，为文件`tool_master_ren_shi_jian_split_data_2026-01-21_alpaca.json`**

### RL数据生成

此步骤可以实现依据数据分配结果，将其中的rl数据处理成训练所需格式。
进入`DramaSR-LRM/training`目录，运行`generate_rl_data.py`，并指定`--input_filename`为Data Split生成的split_data文件，`--dataset_format`为`verl`（暂不支持其他格式），`--series_name`为剧集名，`--val_ratio`为测试集比例（项目中定为0.1）。
此步骤生成的rl数据（分为train和val两个文件）会储存在当前`training`目录。

**项目在`training`目录中提供了生成好的RL数据，为文件`tool_master_zhen_huan_zhuan_split_data_2026-01-22_verl_0.1_train.parquet` `tool_master_zhen_huan_zhuan_split_data_2026-01-22_verl_0.1_val.parquet`**

## 模型训练

### SFT训练

我们使用LLaMA-Factory框架来实现SFT训练。框架位于`DramaSR-LRM/LLaMA-Factory`。
我们自定义了框架中的两处：
1. `DramaSR-LRM/LLaMA-Factory/examples/train_full/speaker_rec.yaml` - 训练参数配置文件
2. `DramaSR-LRM/LLaMA-Factory/data/dataset_info.json` - 训练数据信息索引文件

你需要将使用的sft数据拷贝到`DramaSR-LRM/LLaMA-Factory/data`目录下，并在`dataset_info.json`中定义新的数据集。项目提供了自带数据集`speaker_rec`，对应项目提供的SFT数据。当你创建新的sft数据集时，相较于`speaker_rec`你只需要改动`file_name`参数使其指向你的sft文件，其余参数无须调整。
你需要在`speaker_rec.yaml`中定义基座模型路径、数据集名称（对应`dataset_info.json`中的key）、模型保存路径。你也可以自定义其中的参数。

上述步骤完成后，你应当进入`DramaSR-LRM/LLaMA-Factory`目录，执行`llamafactory-cli train ./examples/train_full/speaker_rec.yaml`开始训练。


### RL训练

我们使用框架verl-tool来实现RL训练。框架位于`DramaSR-LRM/verl-tool`。
我们自定义了框架中的三处：
1. `DramaSR-LRM/verl-tool/examples/train/speaker_rec.sh` - 训练参数配置文件
2. `DramaSR-LRM/verl-tool/verl_tool/servers/tools/speaker_rec_tool.py` - 工具调用逻辑文件
3. `DramaSR-LRM/verl-tool/verl/verl/utils/reward_score/__init__.py` - 奖励函数设计文件
你需要在`speaker_rec.sh`中定义基座模型路径、训练数据文件路径、测试数据文件路径、模型保存路径、训练rollout保存路径、测试rollout保存路径。你也可以自定义其中的参数。目前项目只支持用中文剧集进行RL训练。

上述步骤完成后，你应当进入`DramaSR-LRM/verl-tool`目录，执行`bash ./examples/train/speaker_rec.sh`开始训练。

## 模型评测

最终我们可以使用训练好的模型来在剧集上测试说话人识别效果。
进入`DramaSR-LRM/LRM`目录，在`model_batch.sh`中定义模型路径，并运行`model_batch.sh`脚本。你可以根据环境需求调整其中的GPU数量配置与基础端口等参数，修改后请前往`model_chat.py`检查是否需要同步参数的更改情况。脚本默认会在每个GPU上加载一份模型。
运行`tool_master.py`，并指定`--series_name`为剧集名称，`--end_name`为文件输出名后缀，`--if_api`为`no`，`--port_s`和`--port_e`为可供程序交互的端口偏移区间，为双闭区间。例如，指定`--port_s`为1，`--port_e`为1即代表程序仅与端口号为`<baseport+1>`的端口交互。程序支持多线程并发访问，因此在多端口部署模型与程序交互可以大幅提升评测速度。
生成的评测初始文本会储存在`test_result`文件夹中。

你可以运行`data_collect.py`来对结果进行收集和统计。运行时你需要指定`--series_name`为剧集名称，`--end_name`为与评测时相同的后缀。
收集与统计结果会储存在`test_result/statistic_count`中。

**我们的模型位于路径`DramaSR-LRM/DramaSR-LRM`**
**我们的评测结果位于路径`DramaSR-LRM/LRM/experiment_result`**








