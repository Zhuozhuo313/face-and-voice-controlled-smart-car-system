import Car_move
from TTS import tts

# 常量配置
DEFAULT_SPEED = 50  #默认速度
SPEED_INCREMENT = 10  #速度变化步长
MAX_SPEED = 100    #最大速度
MIN_SPEED = 20      #最小速度

# 初始化速度
speed = DEFAULT_SPEED

# 命令映射
COMMANDS = {
    "前进": ("forward", "正在前进"),
    "后退": ("backward", "正在后退"),
    "左转": ("turn_left", "正在左转"),
    "右转": ("turn_right", "正在右转"),
    "加速": ("increase_speed", "已加速"),
    "减速": ("decrease_speed", "已减速"),
    "停车": ("stop", "已停车")
}

def execute_command(action):
    global speed
    if action == "forward":
        speed=50
        Car_move.forward(speed)
    elif action == "backward":
        speed=50
        Car_move.backward(speed)
    elif action == "turn_left":
        speed=50
        Car_move.turn(0, speed, speed, 0)
    elif action == "turn_right":
        speed=50
        Car_move.turn(speed, 0, 0, speed)
    elif action == "increase_speed":
        speed = min(speed + SPEED_INCREMENT, MAX_SPEED)
    elif action == "decrease_speed":
        speed = max(speed - SPEED_INCREMENT, MIN_SPEED)
    elif action == "stop":
        Car_move.stop()

def car_control(command):
    global speed

    # 截取命令的前5个字符
    truncated_command = command[:5]

    for key, (action, message) in COMMANDS.items():
        if key in truncated_command:
            execute_command(action)
            print(message)
            return

    print("未识别到有效指令\n")
    tts("无效指令")

# 示例调用
if __name__ == "__main__":
    commands = ["前进", "后退", "左转", "右转", "加速", "减速", "停车", "无效指令"]
    for cmd in commands:
        car_control(cmd)





