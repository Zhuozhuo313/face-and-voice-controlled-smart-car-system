import RPi.GPIO as GPIO
import time



#关闭警告,设置引脚为BCM
GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)

#定义超声波测距类
class Ultrasoic_wave:
    def __init__(self):
        self.GPIO_trigger = 20  # 触发引脚
        self.GPIO_receive = 21     # 接收引脚

        GPIO.setup(self.GPIO_trigger, GPIO.OUT)  # 设置为输出
        GPIO.setup(self.GPIO_receive, GPIO.IN)      # 设置为输入

        self.dist_average = 0  # 初始化移动平均距离

    #测距方法实现
    def DistMeasure(self):
        #生成一个10微秒的脉冲信号，用于触发超声波传感器发射超声波。
        GPIO.output(self.GPIO_trigger, GPIO.LOW)
        time.sleep(0.000002)
        GPIO.output(self.GPIO_trigger, GPIO.HIGH)
        time.sleep(0.00001)
        GPIO.output(self.GPIO_trigger, GPIO.LOW)
        
        #根据速度时间公式计算距离
        start_time = time.time() #计时
        while GPIO.input(self.GPIO_receive) == 0:
            if time.time()-start_time >= 0.01: 
                print('超时，未收到回音')
                return 0

        #收到回音，重新计时
        start_time = time.time()

        while GPIO.input(self.GPIO_receive) == 1:
            pass
        stop_time = time.time()
        time_elapsed = stop_time - start_time #记录收到回音的总时长,高电平持续的时间就是超声波从发射到返回的时间
        
        #超声波速度为340m/s
        distance = (time_elapsed * 340) / 2

        return distance
    
    #提升测量准确性，滑动平均测距法
    def DistMeasure_MovingAverage(self):
        dist_current = self.DistMeasure()
        if dist_current == 0:
            return self.dist_average
        else:
            self.dist_average = 0.8*dist_current + 0.2*self.dist_average
            return self.dist_average

if __name__ == '__main__':
    try:
        car = Ultrasoic_wave()
        while True:
            dist = car.disMeasure()
            print("Measured Distance = {:.2f} m".format(dist))
            time.sleep(3)
  
        # Reset by pressing CTRL + C
    except KeyboardInterrupt:
        print("Measurement stopped by User")
        GPIO.cleanup()



