# **Lab2_SSD**

### **a. Качественный анализ работы нейронной сети на 4 снимках**

На 1-ой картинке автомобили и люди, которые чётко видны и не загорожены \
посторонними объектами были достаточно хорошо детектированы\
![image1](https://github.com/temp-rw/Lab2_SSD/blob/master/Images/Pic1.png)

На 2-ой картинке видно, что сеть не справилась с детектированием
людей и транспорта\
![image2](https://github.com/temp-rw/Lab2_SSD/blob/master/Images/Pic2.png)

На 3-ем изображениии сеть снова плохо справилась с детектированием\
людей и автомобилей\
![image3](https://github.com/temp-rw/Lab2_SSD/blob/master/Images/Pic3.png)

На 4-м изображении сеть детектировала только автомобиль, но не обнаружила\
грузовиков, но в целом справилась с детектированием неплохо\

![image4](https://github.com/temp-rw/Lab2_SSD/blob/master/Images/Pic4.png)

### **b. Таблица для п.4**
Средняя точность детектирования объектов, количество пропущенных объектов,\
количество ложных тревог для порогов IoU = 0.5, 0.75, 0.9

![IoU](https://github.com/temp-rw/Lab2_SSD/blob/master/Tables/IoU.png)

### **c. Таблица для п.5**
Средняя точность детектирования объектов, количество пропущенных объектов,\
количество ложных тревог для каждого класса для порогов IoU = 0.5, 0.75, 0.9

1. IoU = 0.5\
![Table50](https://github.com/temp-rw/Lab2_SSD/blob/master/Tables/Table50.png)
2. IoU = 0.75\
![Table75](https://github.com/temp-rw/Lab2_SSD/blob/master/Tables/Table75.png)
3. IoU = 0.9\
![Table90](https://github.com/temp-rw/Lab2_SSD/blob/master/Tables/Table90.png)
