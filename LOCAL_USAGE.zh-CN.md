# 本地使用训练好的 TinyStories 模型

## 启动交互程序

在仓库根目录运行：

```bash
scripts/story.sh
```

首次启动需要加载约 260MiB 的 checkpoint。看到下面的提示后，输入英文故事开头并按回车：

```text
Story opening>
```

例如：

```text
Once upon a time, a little rabbit found a red balloon.
```

模型会在同一终端中续写故事。可以继续输入下一个故事开头，不需要重新加载模型。输入 `quit` 或按 `Ctrl-C` 退出。

## 调整生成长度和随机性

```bash
# 最多生成 300 tokens
scripts/story.sh --max-new-tokens 300

# 更保守、随机性更低
scripts/story.sh --temperature 0.6 --top-p 0.8

# 使用 CPU 而不是 Mac MPS
scripts/story.sh --device cpu
```

## 单次生成

```bash
scripts/story.sh \
  --prompt "One day, a little boy found a magic key." \
  --max-new-tokens 180
```

## 能做什么

这个模型只有约 22.7M 参数，并且只使用 TinyStories 训练，适合续写简单的英文儿童故事。它不是通用聊天模型，不适合知识问答、数学、代码或复杂指令。输出中的 `<|endoftext|>` 表示模型自行结束了故事。
