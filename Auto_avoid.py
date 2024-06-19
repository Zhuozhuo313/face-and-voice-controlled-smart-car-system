import Car_move
from Infrared_ray import Infrared
from Ultrasonic_ranging import Ultrasoic_wave
import time


infrared = Infrared()
ultrasonic = Ultrasoic_wave()
# 自动避障程序
def auto_avoidance():
    [left_obstacle, right_obstacle] = infrared.obstacleMeasure()
    distance = ultrasonic.DistMeasure_MovingAverage()
    
    if distance < 0.2 or left_obstacle == 0 or right_obstacle == 0:  # 距离小于20厘米或红外检测到障碍物
        Car_move.stop()
        time.sleep(0.5)
        
        if left_obstacle == 0 and right_obstacle == 1:
            Car_move.turn(0, 50, 50, 0)  # 左转避开左侧障碍物
            time.sleep(1)
        elif right_obstacle == 0 and left_obstacle == 1:
            Car_move.turn(50, 0, 0, 50)  # 右转避开右侧障碍物
            time.sleep(1)
        else:
            Car_move.backward(50)  # 后退避开前方障碍物
            time.sleep(1)
            Car_move.turn(50, 0, 0, 50)  # 随机右转
            time.sleep(1)
    else:
        Car_move.forward(50)  # 没有障碍物，继续前进
    
    time.sleep(0.1)

if __name__ == "__main__":
    try:
        while True:
            auto_avoidance()
    except KeyboardInterrupt:
        print("\nCaught Ctrl + C. Exiting")








