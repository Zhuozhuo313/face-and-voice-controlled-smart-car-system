print("开机启动，系统正在初始化..........")
from concurrent.futures import ThreadPoolExecutor
from face_tracker import face_track
from Car_control import car_control
from Car_move import stop_all   #停止电机并清理GPIO
from ASR import realTime_ASR  # 语音识别
from VAW import vaw          # 语音唤醒
from TTS import tts         # 语音合成
from Auto_avoid import *
from Car_move import forward as car_forward
from Car_move import backward as car_backward
from Car_move import turn as car_turn
from Car_move import stop as car_stop
import time
from Picamera2_Img_et import Imget
import os
import sys
print("初始化系统完成！")

print("欢迎进入智能小车控制系统\n")
tts("欢迎进入智能小车控制系统")

#是否启动自动避障和目标追踪的标志
auto_avoid_exit=False
tracking_exit=False

#自动避障
def auto_avoid():
    global auto_avoid_exit
    while auto_avoid_exit:
        auto_avoidance()
    car_stop()

#目标追踪
def tracking():
    global tracking_exit
    while tracking_exit:
        car_commend = face_track()
        if car_commend == 1:
            car_turn(50,0,0,50)
            time.sleep(1)
            car_stop()
        elif car_commend == 2:
            car_turn(0,50,50,0)
            time.sleep(1)
            car_stop()
        elif car_commend == 3:
            car_forward(50)
            time.sleep(1)
            car_stop()
        elif car_commend == 4:
            car_backward(50)
            time.sleep(1)
            car_stop()

def main():
    global auto_avoid_exit, tracking_exit
    system_exit = True

    #创建一个线程池方便任务管理
    with ThreadPoolExecutor(max_workers=2) as executor:
        while system_exit:
            #语音唤醒
            while True:   
                name, vaw_result = vaw()
                if name == "LONG" and vaw_result == "小车小车":
                    print("我在，随时为您待命")
                    tts("我在")
                    break
            
            #等待语音指令
            while True:
                command = realTime_ASR()
                if "关闭自动避障" in command:
                    auto_avoid_exit = False
                    print("已退出自动避障模式")
                    tts("已退出自动避障模式")
                elif "自动避障" in command:
                    if not auto_avoid_exit:
                        auto_avoid_exit = True
                        print("开启自动避障模式")
                        tts("开启自动避障模式")
                        executor.submit(auto_avoid)
                elif "退出目标跟踪" in command:
                    tracking_exit = False
                    print("已关闭目标跟踪")
                    tts("已关闭目标跟踪")
                elif "目标跟踪" in command:
                    if not tracking_exit:
                        tracking_exit = True
                        print("开始目标跟踪")
                        tts("开始目标跟踪")
                        executor.submit(tracking)
                elif "退出系统" in command:
                    print("正在退出系统...........")
                    tts("正在退出系统")
                    auto_avoid_exit = False
                    tracking_exit = False
                    print("系统已退出!")
                    tts("系统已退出")
                    system_exit = False
                    break
                elif command in ["长时间未收到指令", "休眠"]:
                    print("进入休眠状态......")
                    tts("进入休眠状态")
                    break
                else:
                    car_control(command)
                    
        #executor.shutdown(wait=True)

# 主程序入口
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Measurement stopped by User")
    finally:
        stop_all()
        print("Cleaning up resources")






