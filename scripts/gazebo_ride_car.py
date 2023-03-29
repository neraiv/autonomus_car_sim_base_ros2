# Hi, I made this beacues i cant use turtlebot teleop_key
# U can drive robot using w a s d
#

import sys

import geometry_msgs.msg
import rclpy

if sys.platform == 'win32':
    import msvcrt
else:
    import termios
    import tty


msg = """
Every key increases the speed 
w -> forward speed increase
s -> backward speed increase
a -> turn left speed incrase
d -> turn right speed increase
f -> Since im no pro i made f to stop car turn.
---------------------------
Moving around:
    w
a   s   d
Press f to set turn = 0.0

Press c to quit
"""

def getKey(settings):
    if sys.platform == 'win32':
        # getwch() returns a string on Windows
        key = msvcrt.getwch()
    else:
        tty.setraw(sys.stdin.fileno())
        # sys.stdin.read() returns a string on Linux
        key = sys.stdin.read(1)
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
    return key


def saveTerminalSettings():
    if sys.platform == 'win32':
        return None
    return termios.tcgetattr(sys.stdin)


def restoreTerminalSettings(old_settings):
    if sys.platform == 'win32':
        return
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)


def vels(speed, turn):
    return 'currently:\tspeed %s\tturn %s ' % (speed, turn)


def main():
    settings = saveTerminalSettings()

    rclpy.init()

    node = rclpy.create_node('teleop_twist_keyboard')
    pub = node.create_publisher(geometry_msgs.msg.Twist, '/umut/cmd_vel', 10)

    speed = 0.0
    turn = 0.0

    try:
        print(msg)
        print(vels(speed, turn))
        while True:
            key = getKey(settings)
            print(key,speed,turn)
            if key == 'w':
                speed = speed + 0.1
            elif key == 's':
                speed = speed - 0.1
            elif key == 'a':
                turn = turn + 0.2
            elif key == 'd':
                turn = turn - 0.2
            elif key == 'f':
                turn = 0.0
            elif key == "c":              
                twist = geometry_msgs.msg.Twist()
                twist.linear.x = 0.0
                twist.angular.z = 0.0
                pub.publish(twist)

                restoreTerminalSettings(settings)
                break

            twist = geometry_msgs.msg.Twist()
            twist.linear.x = speed
            twist.angular.z = turn
            pub.publish(twist)

    except Exception as e:
        print(e)

if __name__ == '__main__':
    main()