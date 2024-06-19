import RPi.GPIO as GPIO
import time

GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)

#定义一个红外线类
class Infrared(object):
    def __init__(self):
        #设置引脚，避障传感器
        self.GPIO_obstacle_right = 19  
        self.GPIO_obstacle_left = 26
        
        #巡线传感器
        self.GPIO_line_left = 16
        self.GPIO_line_right = 12

        GPIO.setup(self.GPIO_obstacle_right, GPIO.IN)
        GPIO.setup(self.GPIO_obstacle_left, GPIO.IN)
        GPIO.setup(self.GPIO_line_left, GPIO.IN)
        GPIO.setup(self.GPIO_line_right, GPIO.IN)
    
    #测量方法返回0表示检测到障碍物
    def obstacleMeasure(self):
        right_obstacle = GPIO.input(self.GPIO_obstacle_right)
        left_obstacle = GPIO.input(self.GPIO_obstacle_left)
        return [left_obstacle, right_obstacle]

    #追踪方法
    def lineMreasure(self):
        left_line = GPIO.input(self.GPIO_line_left)
        right_line = GPIO.input(self.GPIO_line_right)
        return [left_line, right_line]

if __name__ == '__main__':
    try:
        car = Infrared()
        while True:
            [left_obstacle, right_obstacle] = car.obstacledMeasure()
            [left_line, right_line] = car.lineMreasure()
            print(left_line, right_line)  # 输出巡线传感器的值
            time.sleep(1)

    except KeyboardInterrupt:  #ctrl+c 中断
        print("Measurement stopped by User")
        GPIO.cleanup()







