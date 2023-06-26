import subprocess

# 定义要运行的脚本文件名
script_file = "main.py"

# 定义要运行的次数
num_runs = 125

# 循环运行脚本
for i in range(num_runs):
    # 改变 bias 变量的值，可以根据需要进行调整
    bias = i  # 示例中假设 bias 每次递增 1

    # 构造运行脚本的命令行参数，将 bias 作为参数传递给 main.py
    args = ["python", script_file, "--bias", str(bias)]

    # 使用 subprocess 模块运行脚本
    subprocess.run(args)

    # 可以根据需要在每次运行之间添加适当的延迟或其他操作
