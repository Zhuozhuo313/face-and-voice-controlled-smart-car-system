import RPi.GPIO as GPIO
import time

# 关闭GPIO警告
GPIO.setwarnings(False)
# 设置GPIO模式为BCM
GPIO.setmode(GPIO.BCM)

# 电机引脚定义
IN1 = 17 #IN1
IN2 = 18 #IN2
IN3 = 22 #IN3
IN4 = 23 #IN4

# 设置GPIO引脚为输出模式
GPIO.setup(IN1, GPIO.OUT)
GPIO.setup(IN2, GPIO.OUT)
GPIO.setup(IN3, GPIO.OUT)
GPIO.setup(IN4, GPIO.OUT)

# 初始化PWM，频率为500Hz，控制速率
left_forward = GPIO.PWM(IN1, 500)
left_backward = GPIO.PWM(IN2, 500)
right_forward = GPIO.PWM(IN3, 500)
right_backward = GPIO.PWM(IN4, 500)

# 启动PWM，占空比为0
left_forward.start(0)
left_backward.start(0)
right_forward.start(0)
right_backward.start(0)

#speed为占空比
def forward(speed):
    left_forward.ChangeDutyCycle(speed)
    left_backward.ChangeDutyCycle(0)
    right_forward.ChangeDutyCycle(speed)
    right_backward.ChangeDutyCycle(0)

def backward(speed):
    left_forward.ChangeDutyCycle(0)
    left_backward.ChangeDutyCycle(speed)
    right_forward.ChangeDutyCycle(0)
    right_backward.ChangeDutyCycle(speed)

def turn(speed_left_forward=0,speed_left_backward=0,speed_right_forward=0,speed_right_backward=0):
    left_forward.ChangeDutyCycle(speed_left_forward)
    left_backward.ChangeDutyCycle(speed_left_backward)
    right_forward.ChangeDutyCycle(speed_right_forward)
    right_backward.ChangeDutyCycle(speed_right_backward)

def stop():
    left_forward.ChangeDutyCycle(0)
    left_backward.ChangeDutyCycle(0)
    right_forward.ChangeDutyCycle(0)
    right_backward.ChangeDutyCycle(0)

def stop_all():
    left_forward.stop()
    left_backward.stop()
    right_forward.stop()
    right_backward.stop()
    GPIO.cleanup()

if __name__=="__main__":
    try:
        # 前进，速度为50%
        forward(50)
        time.sleep(2)

        forward(80)
        time.sleep(2)

        # 后退，速度为50%
        backward(50)
        time.sleep(2)

        # 左转，速度为50%
        turn(0,50,50,0)
        time.sleep(2)

        # 右转，速度为50%
        turn(50,0,0,50)
        time.sleep(2)

        # 刹车
        stop()
        time.sleep(2)



    finally:
            # 停止所有电机并清理GPIO
            stop_all()


