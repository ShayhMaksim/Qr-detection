# Qr-detection
Был разработан алгоритм для навигации по Qr-коду в помещении.
Qr-код должен содержать в себе следующие данные:
[size z x y angle]

size - реальный размер Qr-кода (в мм.)  
z - высота Qr-кода в локальной системек координат (в мм.)  
x , y - координаты Qr-кода в локальной системек координат (в мм.)  
angle - угол перехода из системый координат Qr-кода в локальную систему.  

zbar-opencv-comparison.py - главный файл.

Для запуска приложения достаточно ввести команду:
python3 zbar-opencv-comparison.py

![Image alt](https://github.com/ShayhMaksim/Qr-detection/blob/main/test/ForGithub.png)

