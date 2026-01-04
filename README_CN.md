## 快速开始
```shell
# 安装deepspeed
git clone https://github.com/gliangMT/DeepSpeed.git
cd DeepSpeed
git checkout musa_dev
DS_ACCELERATOR=musa DS_BUILD_FUSED_ADAM=1 DS_BUILD_CPU_ADAM=1 DS_BUILD_CPU_ADAGRAD=1 python setup.py develop
cd ..

git clone https://github.com/sunyanguomt/Tencent-Hunyuan-Large.git
cd Tencent-Hunyuan-Large
git checkout musa_dev
pip install tiktoken
pip install peft==0.12.0
# 数据集  https://huggingface.co/datasets/a-m-team/AM-DeepSeek-R1-Distilled-1.4M 自行下载，放到train下

# 使用fp8需要accelerate做一些修改具体参考https://github.com/sunyanguomt/accelerate/pull/1, 或直接使用https://github.com/sunyanguomt/accelerate
# 采集profiling需要transformers添加采集功能
```


<p align="left">
   <a href="README.md">English</a>  ｜ 中文</a>&nbsp
</p>
<br><br>

<p align="center">
 <img src="https://dscache.tencent-cloud.cn/upload/uploader/hunyuan-64b418fd052c033b228e04bc77bbc4b54fd7f5bc.png" width="400"/> <br>
</p><p></p>

<p align="center">
    🫣&nbsp<a href="https://huggingface.co/tencent/Tencent-Hunyuan-Large"><b>Hugging Face</b></a>&nbsp&nbsp |  &nbsp&nbsp🖥️&nbsp&nbsp<a href="https://llm.hunyuan.tencent.com/" style="color: red;"><b>官网</b></a>&nbsp&nbsp｜&nbsp&nbsp🕖&nbsp&nbsp <a href="https://cloud.tencent.com/product/hunyuan" ><b>混元API</b></a>｜&nbsp&nbsp🐳&nbsp&nbsp <a href="https://gitee.com/Tencent/Tencent-Hunyuan-Large" ><b>Gitee</b></a>
</p><p align="center">
    <a href="https://arxiv.org/abs/2411.02265" style="color: red;"><b>技术报告</b></a>&nbsp&nbsp｜&nbsp&nbsp <a href="https://huggingface.co/spaces/tencent/Hunyuan-Large"><b>Demo</b></a>&nbsp&nbsp&nbsp｜&nbsp&nbsp <a href="https://cloud.tencent.com/document/product/851/112032" style="color: red;"><b>Tencent Cloud TI</b></a>&nbsp&nbsp&nbsp</p>



## 模型介绍

随着人工智能技术的快速发展，大型语言模型（LLMs）在自然语言处理、计算机视觉和科学任务等领域取得了显著进展。然而，随着模型规模的扩大，如何在保持高性能的同时优化资源消耗成为一个关键挑战。为了应对这一挑战，我们研究了混合专家（MoE）模型，当前亮相的Hunyuan-Large（Hunyuan-MoE-A52B）模型，这是目前业界已经开源的基于Transformer的最大MoE模型，拥有3890亿总参数和520亿激活参数。

本次通过开源Hunyuan-Large的技术成果，我们希望激发更多研究者的创新灵感，共同推动AI技术的进步和应用。欢迎加入我们的开源社区，共同探索和优化未来的AI模型！

### 技术优势介绍

#### 模型  
- **高质量合成数据**：通过合成数据增强训练，Hunyuan-Large能够学习到更丰富的表示，处理长上下文输入，并更好地泛化到未见数据

- **KV缓存压缩**：采用分组查询注意力（GQA）和跨层注意力（CLA）策略，显著减少了KV缓存的内存占用和计算开销，提高了推理吞吐

- **专家特定学习率缩放**：为不同专家设置不同的学习率，确保每个子模型都能有效地从数据中学习，并为整体性能做出贡献

- **长上下文处理能力**：预训练模型支持高达256K的文本序列，Instruct模型支持128K的文本序列，显著提升了长上下文任务的处理能力

- **广泛的基准测试**：在多种语言和任务上进行广泛实验，验证了Hunyuan-Large的实际应用效果和安全性

#### 推理框架
- Hunyuan-Large模型支持 TRT-LLM-backend 和 [vLLM-backend](https://github.com/quinnrong94/vllm/tree/dev_hunyuan) 推理框架。我们在开源框架的基础上适配了Hunyuan-Large模型，譬如，新增的CLA结构可以很大程度节约显存(KV-Cache部分节省50%)，保障超长文本场景。此外通过FP8的量化优化，相比FP16/BF16常规量化，在最大限度保障精度的条件下，节省50%显存，吞吐提升70%。同时，基于TRT-LLM的底层高效算子，其性能相比vLLM提升30%以上，目前TRT-LLM方案在腾讯混元项目广泛使用。本次优先开源vLLM框架，TRT-LLM将在近期推出。

#### 训练框架
- Hunyuan-Large开源模型已经支持huggingface格式，支持用户采用hf-deepspeed框架进行模型精调， 同时我们也支持利用flash-attn进行训练加速，为此，我们把相关的训练脚本和模型实现也开放给到社区，方便研发者在此基础上进行后续的模型训练和精调的操作

&nbsp;

## 新闻
* 2024.11.25 我们自主开发的长上下文评估集——PenguinScrolls，已经正式发布！详见[GitHub](https://github.com/Penguin-Scrolls/PenguinScrolls)和 [Hugging Face](https://huggingface.co/datasets/Penguin-Scrolls/PenguinScrolls)。  
* 2024.11.20 **Hunyuan-A52B-Instruct** 和**Hunyuan-A52B-Instruct-FP8**模型权重更新。
* 2024.11.5 [TI平台](https://cloud.tencent.com/product/ti) 已经集成了Hunyuan-Large模型，您只需几步即可轻松进行训练和部署。访问 [Chat with Hunyuan-Large](https://console.cloud.tencent.com/tione/v2/aimarket/detail/hunyuan_series?PublicAlgoGroupId=hunyuan-large-chat&detailTab=demo) 与模型的实时对话，并在TI上探索 [Hunyuan-Large Best Practice on TI](https://cloud.tencent.com/document/product/851/112032) 并创建自己的定制化Hunyuan-Large。
* 2024.11.5 我们在Hugging Face开源了**Hunyuan-A52B-Pretrain** 、 **Hunyuan-A52B-Instruct** 和**Hunyuan-A52B-Instruct-FP8**。并发布了技术报告和训练推理操作手册，详细介绍了模型能力和训练与推理的操作。
<br>


## Benchmark评估榜单 

**Hunyuan-Large 预训练模型**与具有相似激活参数大小的Dense和MoE竞争对手相比，实现了最佳的整体性能。
对于MMLU、MMLU-pro、CMMLU等基准评测，Hunyuan-Large的性能始终保持在最佳水准，证实了它在聚合任务上的综合能力。
Hunyuan-Large在常识理解和推理以及经典的NLP任务，如QA和阅读理解任务（CommonsenseQA， PIQA，和TriviaQA）方面也表现出色。
在数学能力方面，Hunyuan-Large在GSM8K和Math数学数据集上优于所有基线，在CMATH中文数据集上也取得了最好的成绩。
同时我们观察到Hunyuan-Large在所有中文任务（例如，CMMLU, C-Eval）中实现了整体最佳的性能。


| Model            | LLama3.1-405B | LLama3.1-70B | Mixtral-8x22B | DeepSeek-V2 | Hunyuan-Large |
|------------------|---------------|--------------|---------------|-------------|---------------|
| MMLU             | 85.2          | 79.3         | 77.8          | 78.5        | **88.4**          |
| MMLU-Pro         | **61.6**          | 53.8         | 49.5          | -           | 60.2          |
| BBH              | 85.9          | 81.6         | 78.9          | 78.9        | **86.3**          |
| HellaSwag        | -             | -            | **88.7**      | 87.8        | 86.8          |
| CommonsenseQA    | 85.8          | 84.1         | 82.4          | -           | **92.9**          |
| WinoGrande       | 86.7          | 85.3         | 85.0          | 84.9        | **88.7**          |
| PIQA             | -             | -            | 83.6          | 83.7        | **88.3**          |
| NaturalQuestions | -             | -            | 39.6          | 38.7        | **52.8**          |
| DROP             | 84.8          | 79.6         | 80.4          | 80.1        | **88.9**          |
| ARC-C            | **96.1**          | 92.9         | 91.2          | 92.4        | 95.0          |
| TriviaQA         | -             | -            | 82.1          | 79.9        | **89.2**          |
| CMMLU            | -             | -            | 60.0          | 84.0        | **90.2**          |
| C-Eval           | -             | -            | 59.6          | 81.7        | **91.9**          |
| C3               | -             | -            | 71.4          | 77.4        | **82.3**          |
| GSM8K            | 89.0          | 83.7         | 83.7          | 79.2        | **92.8**          |
| MATH             | 53.8          | 41.4         | 42.5          | 43.6        | **69.8**          |
| CMATH            | -             | -            | 72.3          | 78.7        | **91.3**          |
| HumanEval        | 61.0          | 58.5         | 53.1          | 48.8        | **71.4**          |
| MBPP             | **73.4**          | 68.6         | 64.2          | 66.6        | 72.6          |

**Hunyuan-Large-Instruct**与具有相似激活参数的llm相比在大多数的任务上实现了一致的性能提升，表明我们的post-training十分有效。
在不同类别的基准测试中，我们发现我们的Instruct模型在MMLU和MATH数据集上取得了最好的性能。
值得注意的是，在MMLU数据集上，我们的模型表现出了显著的提升， 相比与LLama3.1-405B模型高出2.6%。
这种增强表明Hunyuan-Large-Instruct在广泛的语言理解任务中具有优越的理解和推理能力。
该模型在MATH数据集上的表现进一步强调了它的实力，相比于LLama3.1-405B高出了3.6%的指标。
值得注意的是，仅用520亿个激活参数就实现了精度的飞跃，证明了Hunyuan-Large-Instruct的卓越能力。

| Model                | LLama3.1 405B Inst. | LLama3.1 70B Inst. | Mixtral 8x22B Inst. | DeepSeekV2.5 Chat | Hunyuan-Large Inst. |
|----------------------|---------------------|--------------------|---------------------|-------------------|---------------------|
| MMLU                 | 87.3                | 83.6               | 77.8                | 80.4              | **89.9**            |
| CMMLU                | -                   | -                  | 61.0                | -                 | **90.4**            |
| C-Eval               | -                   | -                  | 60.0                | -                 | **88.6**            |
| BBH                  | -                   | -                  | 78.4                | 84.3              | **89.5**            |
| HellaSwag            | -                   | -                  | 86.0                | **90.3**          | 88.5                |
| ARC-C                | **96.9**            | 94.8               | 90.0                | -                 | 94.6                |
| GPQA_diamond         | **51.1**            | 46.7               | -                   | -                 | 42.4                |
| MATH                 | 73.8                | 68.0               | 49.8                | 74.7              | **77.4**            |
| HumanEval            | 89.0                | 80.5               | 75.0                | 89.0              | **90.0**            |
| AlignBench           | 6.0                 | 5.9                | 6.2                 | 8.0               | **8.3**             |
| MT-Bench             | 9.1                 | 8.8                | 8.1                 | 9.0               | **9.4**             |
| IFEval strict-prompt | **86.0**            | 83.6               | 71.2                | -                 | 85.0                |
| Arena-Hard |  69.3            | 55.7               |  -                | 76.2                 | **81.8**            |
| AlpacaEval-2.0 | 39.3            | 34.3               | 30.9                | 50.5                 | **51.8**            |

&nbsp;
## 数据

Hunyuan-Large提供了模型训练相关流程，您可以在此章节对训练数据格式进行处理以供模型训练使用。

### 训练数据格式及处理 

训练数据按照以下形式处理为messages格式，训练和推理的默认system prompt为"You are a helpful assistant."，以下分别为单轮数据和多轮数据样例：

```python
# 单轮数据格式
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "海水为什么是咸的" },
    {"role": "assistant", "content": "海水是咸的主要是因为其中含有许多溶解在水中的盐类和矿物质。这些盐类和矿物质来自于地球表面的岩石和土壤中的化学物质，随着时间的推移，它们被带到了海洋中。当海水蒸发时，水分蒸发掉了，但盐类和矿物质仍然留在水中，导致海水变得更加咸味。因此，海水的咸度是由其中的盐类和矿物质的含量决定的。"}
]

# 多轮数据格式
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "知道重庆森林这部电影吗？" },
    {"role": "assistant", "content": "知道呀，是一部由王家卫导演的片子。" },
    {"role": "user", "content": " 而主演里更是有王菲，一上映便受到追捧。" },
    {"role": "assistant", "content": "所以此片获得了第14届香港电影金像奖最佳影片奖。"}
]

from tokenizers import ByteLevelBPETokenizer
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("./models", use_fast=False, trust_remote_code=True)

ids = tokenizer.apply_chat_template(messages)

```

更多调用参考参考`./models/test.py`文件。


&nbsp;

## 快速开始

您可以参照<a href="examples/README.md">快速开始文档</a>中的内容进行快速上手。

## 模型训练 

为了简化部署过程，HunyuanLLM提供了预构建docker镜像：
 [hunyuaninfer/hunyuan-large](https://hub.docker.com/repository/docker/hunyuaninfer/hunyuan-large/general) 。

### 硬件需求

经过在 H20 上测试，不开 make_moe_param_leaf_module 以及 zero3+offload，max_seq_length 为 2048，全量微调最少需要 32 卡，lora 微调最少需要 8 卡。

### 训练性能

最低配置（8 卡 lora 精调）测试下，per_device_train_batch_size 为 1，gradient_accumulation_steps 为 1，大约 35s 一个 iteration。

### 启动方式

参考：[HuggingFace Transformers Trainer](https://huggingface.co/docs/transformers/v4.19.2/en/main_classes/trainer)

#### 单机启动训练

在`train`目录下，执行：

```sh
pip install -r requirements.txt
bash train.sh
```

#### 多机启动训练

如果要用多台机器启动训练，请按照以下步骤执行，并保证多台机器在一个集群内。

##### 配置机器间免密 ssh 登录

以下操作以两个机器为例，两台机器的 ip 分别以`${ip1}`和`${ip2}`标识，以下操作均在 docker container 内执行。

首先，配置多机container免密，在每台机器上执行。

```sh
ssh-keygen			# 生成id_rsa和id_rsa.pub，用于免密登录
ssh-keygen -t rsa -A    # 生成/etc/ssh/ssh_host_rsa_key和ssh_host_ecdsa_key， 用于后面启动ssh listen
/usr/sbin/sshd -p 36005 -o ListenAddress=0.0.0.0        # 启动Listen
echo "Port 36005" > ~/.ssh/config   # ssh 连接端口修改为 36005
passwd root    # 需要配置root密码，否则监测平台会报警
```

注意：这里的`36005`是一个示例端口，可以选用任意端口，但需要保证使用的端口**开放**且**不被其他的进程占用**。

接下来，在每台机器的 container 内，执行：

```sh
cat ~/.ssh/id_rsa.pub
```

**将输出的 ssh 公钥复制并粘贴到`~/.ssh/authorized_keys`文件中，每行一个公钥，每台机器上都要做这个操作**。最终每台机器上的`~/.ssh/authorized_keys`文件内容应当是一致的，并且包含了所有机器的公钥。

需要注意，多节点训练时，每个节点上执行的代码都得一致，建议挂载一个共享的网络盘，如果无法挂载共享网盘，则需要手动将数据集、脚本、代码复制在多台机器的相同目录下。

##### 启动多机训练

在以上准备步骤准备好了之后，以及确认依赖已经安装完成（如未安装，请执行`pip install -r requirements.txt`安装），就可以在`train.sh`中的开头增加以下配置：

```shell
export HOST_GPU_NUM=8
# 当前机器ip
export LOCAL_IP=${ip1}
# 多节点机器ip，逗号隔开
export NODE_IP_LIST="${ip1}:8,${ip2}:8"
# 机器节点个数
export NODES=2
export NODE_NUM=$((${NODES} * ${HOST_GPU_NUM}))
```

注意：将以上的`${ip1}`和`${ip2}`替换为真实的 ip 地址！

然后，在`${ip1}`的机器上，在`train/`目录下，执行`bash train.sh`即可，注意第一次启动时可能会看见以下的输出：

```ssh
The authenticity of host '[ip]:36005 ([ip]:36005)' can't be established.
ECDSA key fingerprint is xxxxxx.
ECDSA key fingerprint is MD5:xxxxxx.
Are you sure you want to continue connecting (yes/no)?
```

此时输入`yes`即可继续。

##### 关键参数

脚本中的关键参数如下：

- `--deepspeed`: 此参数应当指向一个 deepspeed 的配置文件，`train`文件夹下提供了三种 DeepSpeed 的默认配置文件：`ds_zero2_no_offload.json`, `ds_zero3_no_offload.json`, `ds_zero3_offload.json`，这三个配置文件所需显存依次减少
- `--model_name_or_path`: 要加载的 HF 预训练模型权重，确保这个路径下包含了 `modeling_hunyuan.py` 和 `configuration_hunyuan.py` 文件，否则无法加载
- `--tokenizer_name_or_path`: tokenizer 文件夹路径，确保这个路径下包含了`tokenization_hy.py` 文件，否则无法加载
- `--train_data_file`: 训练文件路径，应该为一个 jsonl 文件
- `--output_dir`: 输出文件夹，log、tensorboard 和权重都会存储在这个路径下
- `--per_device_train_batch_size`: 每张卡上的 batch size
- `--gradient_accumulation_steps`: 梯度累计次数，`per_device_train_batch_size * gradient_accumulation_steps * dp_size`为 global_batch_size
- `--max_steps`: 训练的总步数
- `--save_steps`: 每多少个 step 存储一个 checkpoint
- `--use_lora`: 是否用 lora 训练，同时接收`--lora_rank`，`--lora_alpha`和`--lora_dropout`参数。lora 默认应用于 "q_proj", "k_proj", "v_proj", "o_proj" 四个参数，如果需要改变的话在代码中修改即可。注意：**使用 lora 训练时，只会保存 lora 的权重，而不会保存 base 模型的权重**，如果需要合并 lora 权重，看下面的“Lora 权重合并”一节
- `--make_moe_param_leaf_module`：当用 zero3 以及 MoE 训练时，将 MoE 模块视作一个 leaf module，即它的参数不进行 zero3 切分，这个选项预计会显著增加显存占用
- `--gradient_checkpointing`：开启梯度重计算
- `--train_attention_params_only`: 是否只训练 attention 参数
- `--learning_rate`: 训练时的最大学习率
- `--min_lr`: 训练时的最小学习率
- `--use_flash_attn`: 开启 flash-attention 进行训练加速

**注意：**

- 如果想从一个中途保存的 ckpt 继续训练，而不是加载一个预训练的权重，直接指定`--resume_from_checkpoint`为之前训练保存的 ckpt 路径，不要指定`--model_name_or_path`，这样只会加载权重，而不会加载训练状态
- 从 ckpt 继续训练时，loss 可能会有微小的偏差，这是由一些非确定性算法带来的随机性，是正常现象。参考：[HuggingFace Transformers Trainer Randomness 
- 当 `--model_name_or_path` 有效时，所有模型相关的参数都会被忽略
- 一个 batch 内的样本会通过 padding 对齐 batch 内最长的样本，而每条样本的长度最长为 max_seq_length，超出的部分会被裁剪
- 如果报出 bias 权重没有 load 的 warning，忽略即可，Hunyuan-Large 中不会用到 bias

#### 显存不足怎么办？

参考：[DeepSpeed Configuration](https://www.deepspeed.ai/docs/config-json/)

可以尝试修改 ds config，去掉这几个参数的 auto 属性，改小试试看：

- `stage3_param_persistence_threshold`
- `stage3_prefetch_bucket_size`
- `stage3_max_reuse_distance`
- `stage3_max_reuse_distance`


#### Lora 模型合并

保存下来的 lora 权重没法在训练运行时合并到 zero3 模型中，因为 zero3 开启时模型权重会切分到各 dp rank 上。因此如果想把 lora 权重合并到 base 模型上，可以通过离线的方式合并后得到权重文件。执行`merge_lora_weight.sh`即可完成 lora 权重和 base 模型权重的合并，其中的参数有：

- `--base_model_path`：base 模型的权重目录
- `--adapter_model_path`：lora 权重目录
- `--output_path`：合并后的权重保存目录
- `--save_dtype`： 以什么数据格式存储合并后的权重，可选值：fp16，bf16，fp32

&nbsp;

## 推理和部署 

HunyuanLLM支持TRT-LLM和vLLM两种部署方式。本次我们开源vLLM部署方式(详见'使用vLLM推理'章节)，TRT-LLM部署方式(详见'使用TRT-LLM推理'章节)将在近期开放。

## 使用TRT-LLM推理
待开放

## 使用vLLM推理
### Docker:

为了简化部署过程，HunyuanLLM提供了预构建docker镜像：

 [hunyuaninfer/hunyuan-large](https://hub.docker.com/repository/docker/hunyuaninfer/hunyuan-large/general) 。您只需要下载模型文件并用下面代码启动docker即可开始推理模型。
```shell
docker run --name hunyuanLLM_infer -itd --privileged --user root  --net=host --ipc=host --gpus=8 hunyuaninfer/hunyuan-large:infer-open-source
```

注: Docker容器权限管理。以上代码采用特权模式（--privileged）启动Docker容器会赋予容器较高的权限，增加数据泄露和集群安全风险。建议在非必要情况下避免使用特权模式，以降低安全威胁。对于必须使用特权模式的场景，应进行严格的安全评估，并实施相应的安全监控、加固措施。


### 配置机器间免密 ssh 登录

以下操作以两个机器为例，两台机器的 ip 分别以`${ip1}`和`${ip2}`标识，以下操作均在 docker container 内执行。

首先在两台机器上面运行：`passwd`设置密码，例如：`Tmp123,./`

将`inference/login_ssh.py`拷贝到容器中，执行如下命令，注意IP和密码填入正确值。

```shell
python3 login_ssh.py --ips ${ip1},${ip2} --port 36000 --password=Tmp123,./
```

**注意📢：在启动前请务必通过VLLM的多机验证脚本:https://docs.vllm.ai/en/latest/getting_started/debugging.html**

### BF16部署

BF16需要16卡H20部署。验证多机通信无误后，按如下步骤执行：

运行命令前请先设置如下环境变量：

```shell
${LOCAL_IP}：当前机器bond1对应IP
${MODEL_PATH}：Hunyuan LLM模型路径
```

#### Step1：Ray启动

Ray 是一个并行和分布式 Python 的开源库，本章节我们采用Ray来实现多机通信。

Ray组件配置加固：Ray组件默认配置中服务端口（如6379、8265）未启用身份验证机制，存在未授权访问和命令执行的风险。建议在部署Ray组件时，仅在受信任的内部网络环境中进行，或确保对这些端口实施严格的访问控制列表（ACL）策略，禁止非授权网络访问。

首先我们在各个节点上启动ray（放在后台启动或者保持终端运行状态）:

主节点上：
```shell
export VLLM_HOST_IP=${LOCAL_IP}
export NCCL_SOCKET_IFNAME=bond1
export GLOO_SOCKET_IFNAME=bond1
ray start --block  --head --node-ip-address=${LOCAL_IP} --port=6379
```

所有子节点：

注意：{主节点$LOCAL_IP}需填入主节点的${LOCAL_IP}
```shell
export VLLM_HOST_IP=${LOCAL_IP}
export NCCL_SOCKET_IFNAME=bond1
export GLOO_SOCKET_IFNAME=bond1
ray start --block --address={主节点$LOCAL_IP}:6379  --node-ip-address=${LOCAL_IP} 
```
如果启动ray失败，执行`ray stop`后再次执行上述命令。


#### Step2：执行推理

#### 方式1：命令行推理

下面我们展示一个代码片段，采用`vLLM`快速请求chat model：

注: vLLM组件远程代码执行防护。下列代码中vLLM组件的trust-remote-code配置项若被启用，将允许加载并执行来自远程模型仓库的代码，这可能导致恶意代码的执行。除非业务需求明确要求，否则建议该配置项处于禁用状态，以降低潜在的安全威胁。


```python
import os
from vllm import LLM, SamplingParams

model_path=os.environ.get('MODEL_PATH')

llm = LLM(model=model_path,
        tokenizer=model_path,
        trust_remote_code=True,
        max_model_len=10240,
        dtype='bfloat16',
        tensor_parallel_size=16,
        pipeline_parallel_size=1,
        disable_log_stats=False,
        gpu_memory_utilization=0.98,
        disable_custom_all_reduce=True,
        #distributed_executor_backend='ray',
        enforce_eager=True,
        max_num_seqs=8,
        use_v2_block_manager=True,
        quantization=None)

prompts = ["海水为什么是咸的"]

sampling_params = SamplingParams(
    temperature=0.7, top_p=0.6, max_tokens=200, top_k=20, repetition_penalty=1.05)

outputs = llm.generate(prompts, sampling_params)

# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
```

#### 方式2：服务化推理

下面我们展示使用`vLLM`服务化的方式部署模型并请求

在主节点上运行：

```shell
export VLLM_HOST_IP=${LOCAL_IP}
export NCCL_SOCKET_IFNAME=bond1
export GLOO_SOCKET_IFNAME=bond1
```
接着我们启动服务，运行 :
```shell
cd inference
sh run_server.sh
```

*Tips*：故障处理，如果遇到
```python
ray, exceptions.RaySystemError: System error: No module named 'transformers_modules' traceback: Traceback (most recent call last):
ModuleNotFoundError: No module named 'transformers modules'
```
将主节点上的 ~/.cache/huggingface/modules/拷贝到所有子节点相应路径。

运行`run_server.sh`成功后, 运行请求脚本：
```shell
sh openapi.sh
```

注意修改`openapi.sh`中的`${LOCAL_IP}`和`${MODEL_PATH}`为服务对应值。


### 量化模型部署：

本部分介绍采用vLLM部署量化后模型的流程。

镜像：部署镜像同BF16。


#### Int8量化模型部署：
部署Int8-weight-only版本Hunyuan-L模型只需设置`run_server_int8.sh`中的环境变量：
```SHELL
${MODEL_PATH} : BF16模型路径
${LOCAL_IP} : 当前机器bond1对应IP
```

接着我们启动Int8服务。运行：
```shell
sh run_server_int8.sh
```

运行`run_server_int8.sh`成功后, 运行请求脚本：
```shell
sh openapi.sh
```

#### FP8量化模型部署：
部署W8A8C8版本Hunyuan-L模型只需设置`run_server_int8.sh`中的环境变量：
```shell
${MODEL_PATH} : FP8模型路径
${LOCAL_IP} : 当前机器bond1对应IP
```

接着我们启动FP8服务。运行：
```shell
sh run_server_fp8.sh
```

运行`run_server_fp8.sh`成功后, 运行请求脚本：
```shell
sh openapi.sh
```

#### FP8 BENCHMARK

本部分介绍Hunyuan Large Instruct FP8量化模型的效果评估。

| Dataset | BF16 | W8A8C8-FP8 |
|---------|------|------------|
| ARC-C   | 94.6 | 94.2       |
| C-Eval  | 88.6 | 89.2       |
| CMMLU   | 90.4 | 89.8       |
| MMLU    | 89.9 | 88.9       |

### 性能评估：

本部分介绍采用vLLM部署各个模型（原始模型和量化模型）的效率测试结果，包括不同Batchsize下的推理速度(tokens/s)。

| Inference Framework | Model                                                                                                  | Number of GPUs (H20) | input_length | batch=1 | batch=4 |
| ------------------- | ------------------------------------------------------------------------------------------------------ | -------------------- | ------------ |---------|---------|
| vLLM                | Hunyuan-Large                                                                                              | 16                   | 2048         | 20.2    | 75.5    |
| vLLM                | Hunyuan-Large(int8 weight only)                                                                            | 8                    | 2048         | 19.3    | 73.6    |
| vLLM                | Hunyuan-Large(W8A8C8-FP8)                                                                                  | 8                    | 2048         | 19.8    | 74.9    |
## Tokenizer

HunYuan-Large模型中采用的tokenizer平衡了压缩率和效果两个因素，保证embedding可以得到充分的训练。词表包含了从tiktoken中集成的100K个token，在此基础上，我们使用大量的优质中文训练数据，训练了额外的29K的中文token，以增强模型的中文能力和tokenizer对文本的压缩率，二者结合后，与LLaMA3分词器相比，我们的新分词器在压缩率上有所改善，从2.78个字符/token提高到3.13个字符/token。


## 混元API
您可以在腾讯云体验我们的hunyuan-large模型，具体请见：https://cloud.tencent.com/document/product/1729/97730。

## 交互式Demo Web 
Hunyuan-Large现已开放网页demo。访问 https://huggingface.co/spaces/tencent/Hunyuan-Large 即可简单体验我们的模型。

<br>

## 使用TI训练/推理 
腾讯云的 [TI平台](https://cloud.tencent.com/product/ti) 是专门为AI工程师设计的全面的机器学习平台。通过集成Hunyuan-Large模型，您只需几步即可轻松进行训练和部署。访问 [Chat with Hunyuan-Large](https://console.cloud.tencent.com/tione/v2/aimarket/detail/hunyuan_series?PublicAlgoGroupId=hunyuan-large-chat&detailTab=demo) 模块，体验与模型的实时对话，并在TI上探索 [Hunyuan-Large Best Practice](https://cloud.tencent.com/document/product/851/112032) ，创建自己的定制Hunyuan-Large模型。

## 引用
如果你觉得我们的工作对你有帮助，欢迎引用我们！

```
@misc{sun2024hunyuanlargeopensourcemoemodel,
      title={Hunyuan-Large: An Open-Source MoE Model with 52 Billion Activated Parameters by Tencent}, 
      author={Xingwu Sun and Yanfeng Chen and Yiqing Huang and Ruobing Xie and Jiaqi Zhu and Kai Zhang and Shuaipeng Li and Zhen Yang and Jonny Han and Xiaobo Shu and Jiahao Bu and Zhongzhi Chen and Xuemeng Huang and Fengzong Lian and Saiyong Yang and Jianfeng Yan and Yuyuan Zeng and Xiaoqin Ren and Chao Yu and Lulu Wu and Yue Mao and Tao Yang and Suncong Zheng and Kan Wu and Dian Jiao and Jinbao Xue and Xipeng Zhang and Decheng Wu and Kai Liu and Dengpeng Wu and Guanghui Xu and Shaohua Chen and Shuang Chen and Xiao Feng and Yigeng Hong and Junqiang Zheng and Chengcheng Xu and Zongwei Li and Xiong Kuang and Jianglu Hu and Yiqi Chen and Yuchi Deng and Guiyang Li and Ao Liu and Chenchen Zhang and Shihui Hu and Zilong Zhao and Zifan Wu and Yao Ding and Weichao Wang and Han Liu and Roberts Wang and Hao Fei and Peijie She and Ze Zhao and Xun Cao and Hai Wang and Fusheng Xiang and Mengyuan Huang and Zhiyuan Xiong and Bin Hu and Xuebin Hou and Lei Jiang and Jiajia Wu and Yaping Deng and Yi Shen and Qian Wang and Weijie Liu and Jie Liu and Meng Chen and Liang Dong and Weiwen Jia and Hu Chen and Feifei Liu and Rui Yuan and Huilin Xu and Zhenxiang Yan and Tengfei Cao and Zhichao Hu and Xinhua Feng and Dong Du and Tinghao She and Yangyu Tao and Feng Zhang and Jianchen Zhu and Chengzhong Xu and Xirui Li and Chong Zha and Wen Ouyang and Yinben Xia and Xiang Li and Zekun He and Rongpeng Chen and Jiawei Song and Ruibin Chen and Fan Jiang and Chongqing Zhao and Bo Wang and Hao Gong and Rong Gan and Winston Hu and Zhanhui Kang and Yong Yang and Yuhong Liu and Di Wang and Jie Jiang},
      year={2024},
      eprint={2411.02265},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2411.02265}, 
}
```
<br>

## 联系我们
如果你想给我们的研发和产品团队留言，欢迎联系我们腾讯混元LLM团队。你可以通过邮件（hunyuan_opensource@tencent.com）联系我们。
