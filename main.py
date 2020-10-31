import cv2
import math
import numpy as np

#Маштабирование изображение для проверки на цвета(влияет на скорость работы)
colorAccuracy = 0.1
#Адрес видео
video = "res/sample-6.mp4"
#Видеорезультат
outname = "out.avi"
#Цвет зараженного шара (BGR)
redcol = (47, 25, 223)

#Находит точку равноотдаленную от 4 точек
def findCenterm1(top,bot,left,right):
    def findCenterm1_for3(p1,p2,p3):
        x1 = p1[0]
        y1 = p1[1]
        x2 = p2[0]
        y2 = p2[1]
        x3 = p3[0]
        y3 = p3[1]
        a1 = (x2 - x1)
        a2 = (y2 - y1)
        a3 = (x3 - x1)
        a4 = (y3 - y1)
        res1 = x2 *  x2 - x1 * x1 + y2 * y2 - y1 * y1
        res2 = x3 *  x3 - x1 * x1 + y3 * y3 - y1 * y1
    xt = float(top[0])
    yt = float(top[1])
    xb = float(bot[0])
    yb = float(bot[1])
    xl = float(left[0])
    yl = float(left[1])
    xr = float(right[0])
    yr = float(right[1])
    res1 = xr*xr - xl*xl + yr*yr - yl*yl
    res2 = xt*xt - xb*xb + yt*yt - yb*yb
    a1 = 2*(xr - xl)
    a2 = 2*(yr - yl)
    a3 = 2*(xt - xb)
    a4 = 2*(yt - yb)
    if(a1 == 0 or a2 == 0 or a3 == 0 or a4 == 0 ):
        findCenterm1_for3(left,right,top)
        if(a1 == 0 or a2 == 0 or a3 == 0 or a4 == 0 ):
            findCenterm1_for3(left,right,bot)
            if(a1 == 0 or a2 == 0 or a3 == 0 or a4 == 0 ):
                findCenterm1_for3(left,top,bot)
                if(a1 == 0 or a2 == 0 or a3 == 0 or a4 == 0):
                    findCenterm1_for3(right,top,bot)
                    if(a1 == 0 or a2 == 0 or a3 == 0 or a4 == 0 ):

                        x = int(abs((xl+xr))/2)
                        y = int(abs(yt + yb)/2)
                        return [x,y]

    y = ((res2/a3) - (res1/a1))/((a4/a3) - (a2/a1))

    x = (res2 - a4 * y)/a3
    return [int(x),int(y)]

def findCenterm1(top,bot,left,right):
    top = [float(top[0]),float(top[1])]
    bot = [float(bot[0]), float(bot[1])]
    left = [float(left[0]), float(left[1])]
    right = [float(right[0]), float(right[1])]
    def findCenterm1_for3(p1,p2,p3):
        x1 = p1[0]
        y1 = p1[1]
        x2 = p2[0]
        y2 = p2[1]
        x3 = p3[0]
        y3 = p3[1]
        a1 = (x2 - x1)
        a2 = (y2 - y1)
        a3 = (x3 - x1)
        a4 = (y3 - y1)
        res1 = x2 * x2 - x1 * x1 + y2 * y2 - y1 * y1
        res2 = x3 * x3 - x1 * x1 + y3 * y3 - y1 * y1
    xt = (top[0])
    yt = (top[1])
    xb = (bot[0])
    yb = (bot[1])
    xl = (left[0])
    yl = (left[1])
    xr = (right[0])
    yr = (right[1])
    res1 = xr*xr - xl*xl + yr*yr - yl*yl
    res2 = xt*xt - xb*xb + yt*yt - yb*yb
    a1 = 2*(xr - xl)
    a2 = 2*(yr - yl)
    a3 = 2*(xt - xb)
    a4 = 2*(yt - yb)
    if(a1 == 0 or a2 == 0 or a3 == 0 or a4 == 0 ):
        findCenterm1_for3(left,right,top)
        if(a1 == 0 or a2 == 0 or a3 == 0 or a4 == 0 ):
            findCenterm1_for3(left,right,bot)
            if(a1 == 0 or a2 == 0 or a3 == 0 or a4 == 0 ):
                findCenterm1_for3(left,top,bot)
                if(a1 == 0 or a2 == 0 or a3 == 0 or a4 == 0):
                    findCenterm1_for3(right,top,bot)
                    if(a1 == 0 or a2 == 0 or a3 == 0 or a4 == 0 ):
                        x = int(abs((xl+xr))/2)
                        y = int(abs(yt + yb)/2)
                        return [x,y]

    if ((a4/a3) - (a2/a1)) == 0:
        return  [0, 0]
    y = ((res2/a3) - (res1/a1))/((a4/a3) - (a2/a1))

    x = (res2 - a4 * y)/a3
    return [int(x),int(y)]

angle = [[1, 0], [0, -1], [-1, 0], [0, 1]]

#Абсолютна разница 2 чисел
def delta(a1, a2):
    return abs(a1 - a2)

#Отрисовка точки в заданых кординатах
def showP(img, x, y):
    cv2.circle(img, (math.ceil(x), math.ceil(y)), 5, (255, 255, 0), -1)
#Максимальная разница чисел в масиве с числом et
def maxDelta(arr, et):
    arr.sort()
    res = 0;
    s = math.ceil(len(arr) * 0.1)
    if s == 0 or et == 0:
        return 1
    for i in range(s):
        res += delta(arr[i], et)
    res = res / s / et
    return res/et
#Нахождение радиуса по 2 точкам
def radius(p1, p2, an):
    ang = angle[an]
    x = p2[0] - p1[0]
    y = p2[1] - p1[1]
    if x == 0:
        x = 0.001
    if y == 0:
        y = 0.001
    return abs((x*x+y*y)/2/(ang[0] * x + ang[1]*y))
#Закругление масива
def get(arr, i):
    return arr[(len(arr) + i)%len(arr)]
#Растояние между 2 точками
def distance(p1, p2):
    return math.sqrt(math.pow(p1[0]-p2[0], 2) + math.pow(p1[1]-p2[1], 2))
#Нахождение радиуса по контуру
def rad2(c , j, an):
    p = c[j]
    a = math.ceil(len(c)/5)
    if len(c) <= j + a:
        a = len(c) - j - 1
    ra = []
    for k in range(5, a, 5):
        ra.append(distance(p[0],findCenterm1(p[0], get(c, j + k)[0],  get(c, j - k)[0],  get(c, j + math.ceil(k/2))[0])))
    if len(ra) == 0:
        return True, "", ""
    r = sum(ra) / len(ra)
    return False, r, maxDelta(ra, r)
#Изменение размера изображения
def resize(img, k):
    h, w = img.shape[:2]
    dim = (round(w * k), round(h * k))
    return cv2.resize(img, dim, cv2.INTER_AREA)
#Нахождение радиуса по контуру
def rad(c , j, an):
    p = c[j]
    a = math.ceil(len(c)/6)
    if len(c) <= j + a:
        a = len(c) - j - 1
    ra = []
    for k in range(5, a, 5):
        ra.append(radius(p[0], c[j + k][0], an))
    if len(ra) == 0:
        return True, "", ""
    r = sum(ra) / len(ra)
    return False, r, maxDelta(ra, r)
#Среднее значение масива
def av(arr ):
    return sum(arr)/len(arr)
#Ориентация точки
def isOrientated(c, p, an):
    x, y, w, h = cv2.boundingRect(c)
    return abs(p[0][0] * angle[an][0]) == x + int(abs(w * (angle[an][0] + 1) / 2)) - int((angle[an][0] + 1) / 2) or abs(p[0][1] * angle[an][1]) == y + int(abs(h * (angle[an][1] - 1) / 2) + (angle[an][1] - 1) / 2)
#Возвращает мяч по даному контуру
def getBall1(c):
    x, y, w, h = cv2.boundingRect(c)
    cv2.rectangle(img, (x, y), (x + w, y + h), (155, 155, 0), 3)
    xc = x + w / 2
    yc = y + h / 2
    r = 0
    if h / w <= 1:
        r = w / 2
    else:
        r = h / 2
    a = math.ceil(len(c) / 7)
    if w / h < 0.8:
        yr = []
        yl = []
        xb = []
        xt = []
        rp = []
        lp = []
        for j in range(len(c)):
            p = c[j]
            if isOrientated(c, p, 0):
                yr.append(p[0][1])
                if len(rp) == 0:
                    rp.append(get(c, j + a)[0])
                    rp.append(get(c, j + math.ceil(a / 2))[0])
                    rp.append(get(c, j - math.ceil(a / 2))[0])
            if isOrientated(c, p, 2):
                yl.append(p[0][1])
                if len(lp) == 0:
                    lp.append(get(c, j - a)[0])
                    lp.append(get(c, j + math.ceil(a / 2))[0])
                    lp.append(get(c, j - math.ceil(a / 2))[0])
            if isOrientated(c, p, 1):
                xb.append(p[0][0])
            if isOrientated(c, p, 3):
                xt.append(p[0][0])
        yr = av(yr)
        yl = av(yl)

        xb = av(xb)
        xt = av(xt)
        xo = av([xb, xt])
        dr = delta(xo, x)
        dl = delta(xo, x + w)
        if dr > dl:
            xc += r - w / 2
            res = findCenterm1([xc + r, yr], rp[0], rp[1], rp[2])
            xc = res[0]
            yc = res[1]
            r = distance(res, rp[0])
        else:
            xc -= r - w / 2
            res = findCenterm1([xc - r, yl], lp[0], lp[1], lp[2])
            xc = res[0]
            yc = res[1]
            r = distance(res, lp[0])
    if h / w < 0.8:
        xb = []
        xt = []
        yr = []
        yl = []
        bp = []
        tp = []
        for j in range(len(c)):
            p = c[j]
            if isOrientated(c, p, 1):
                xb.append(p[0][0])
                if len(bp) == 0:
                    bp.append(get(c, j - a)[0])
                    bp.append(get(c, j + math.ceil(a / 2))[0])
                    bp.append(get(c, j - math.ceil(a / 2))[0])
            if isOrientated(c, p, 3):
                xt.append(p[0][0])
                if len(tp) == 0:
                    tp.append(get(c, j + a)[0])
                    tp.append(get(c, j + math.ceil(a / 2))[0])
                    tp.append(get(c, j - math.ceil(a / 2))[0])
            if isOrientated(c, p, 0):
                yr.append(p[0][1])
            if isOrientated(c, p, 2):
                yl.append(p[0][1])
        xb = av(xb)
        xt = av(xt)

        yr = av(yr)
        yl = av(yl)
        yo = av([yr, yl])
        db = delta(yo, y + h)
        dt = delta(yo, y)
        if db < dt:
            yc += r - h / 2
            res = findCenterm1([xb, yc + r], bp[0], bp[1], bp[2])
            xc = res[0]
            yc = res[1]
            r = distance(res, bp[0])
        else:
            yc -= r - h / 2
            res = findCenterm1([xc, yc - r], tp[0], tp[1], tp[2])
            xc = res[0]
            yc = res[1]
            r = distance(res, tp[0])
    xc = math.ceil(xc)
    yc = math.ceil(yc)
    r = math.ceil(r)
    return Ball(xc, yc, r)
#Возвращает мяч по даному контуру
def getBall(c):
    x, y, w, h = cv2.boundingRect(c)
    #v2.rectangle(img, (x, y), (x + w, y + h), (155, 155, 0), 3)
    xc = x + w / 2
    yc = y + h / 2
    r = 0
    if h / w <= 1:
        r = w / 2
    else:
        r = h / 2
    if w / h < 0.8:
        yr = []
        yl = []
        xb = []
        xt = []
        for j in range(len(c)):
            p = c[j]
            if isOrientated(c, p, 0):
                yr.append(p[0][1])
            if isOrientated(c, p, 2):
                yl.append(p[0][1])
            if isOrientated(c, p, 1):
                xb.append(p[0][0])
            if isOrientated(c, p, 3):
                xt.append(p[0][0])
        yr = av(yr)
        yl = av(yl)

        xb = av(xb)
        xt = av(xt)
        xo = av([xb, xt])
        dr = delta(xo, x)
        dl = delta(xo, x + w)
        if dr > dl:
            xc += r - w / 2
        else:
            xc -= r - w / 2
    if h / w < 0.8:
        xb = []
        xt = []
        yr = []
        yl = []
        for j in range(len(c)):
            p = c[j]
            if isOrientated(c, p, 1):
                xb.append(p[0][0])
            if isOrientated(c, p, 3):
                xt.append(p[0][0])
            if isOrientated(c, p, 0):
                yr.append(p[0][1])
            if isOrientated(c, p, 2):
                yl.append(p[0][1])
        xb = av(xb)
        xt = av(xt)

        yr = av(yr)
        yl = av(yl)
        yo = av([yr, yl])
        db = delta(yo, y + h)
        dt = delta(yo, y)
        if db < dt:
            yc += r - h / 2
        else:
            yc -= r - h / 2
    xc = math.ceil(xc)
    yc = math.ceil(yc)
    r = math.ceil(r)
    return Ball(xc, yc, r)
#Являеться мячем
def isBall(c):
    correct = False
    for j in range(len(c)):
        p = c[j]
        for an in range(4):
            if isOrientated(c, p, an):
                error, r, md = rad2(c, j, an)
                if error:
                    continue
                if r < 15:
                    continue
                if md < 1:
                    correct = True
    return correct
#Разница цветов
def deltaCol(col1, col2):
    delta =  []
    for i in range(3):
        delta.append(abs(col1[i]- col2[i]))
    return av(delta)


class Ball:

    def __init__(self, x, y, r):
        self.x = x
        self.y = y
        self.r = r
        self.id = None
        self.col = None
        self.healthy = True

    def prams(self):
        return self.x, self.y, self.r

    def isBall(self, ball):
        if deltaCol(self.col, ball.col) <=35:
            return True
        return False

#Возвращает контура мячей
def getContours(img):
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask_1 = cv2.inRange(img_hsv, (0,0,0) , (180, 255, 170) )
    mask_2 =cv2.inRange(img_hsv, (0,0,0) , (180, 150, 255) )
    mask =  cv2.bitwise_or(mask_1, mask_2)
    mask = cv2.bitwise_not(mask)

    img = cv2.bitwise_and(img, img, mask =mask)


    h, w = img.shape[:2]
    k = 0.5
    dim = (round(w * k), round(h * k))
    dim = cv2.resize(img, dim, cv2.INTER_AREA)
    #cv2.imshow("mask", dim)
    #cv2.waitKey(8)

    return cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#Средний цвет
def avCol(img):

    img = resize(img, colorAccuracy)

    average = img.mean(axis=0).mean(axis=0)

    pixels = np.float32(img.reshape(-1, 3))

    n_colors = 5
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
    flags = cv2.KMEANS_RANDOM_CENTERS

    _, labels, palette = cv2.kmeans(pixels, n_colors, None, criteria, 10, flags)
    _, counts = np.unique(labels, return_counts=True)

    dominant = palette[np.argmax(counts)]
    return (round(dominant[0]), round(dominant[1]), round(dominant[2]))
#Получить масив мячей с картинки
def getBalls(img):
    res = []
    contours, hierarchy = getContours(img)
    for c in contours:
        x,y,w,h = cv2.boundingRect(c)
        if len(c) > 50:
            if isBall(c):
                bal = getBall(c)
                bal.col = avCol(img[y:y+h, x:x+w])
                res.append(bal)
    return res
#Обвести круги с масива
def encircle(img, balls):
    for b in balls:
        xc, yc, r = b.prams()
        cv2.circle(img, (xc, yc), r, (255, 100, 0), 2)
        cv2.putText(img, str(b.id), (xc, yc+10), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 0, 150), 1, cv2.LINE_AA)




cap = cv2.VideoCapture(video)
fps = cap.get(cv2.CAP_PROP_FPS)
size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(outname, fourcc, fps, size)
balls = []

red = Ball(-1, -1, 0)
red.col = redcol

nored = True

while (True):
    ret, img = cap.read()
    if(ret == False):
        break

    ab = getBalls(img)

    if len(balls) == 0 :
        for i in range(len(ab)):
            ab[i].id = i + 1
            if nored and ab[i].isBall(red):
                ab[i].healthy = False
                nored = False
            balls.append(ab[i])
    else:
        for i in range(len(ab)):
            for j in range(len(balls)):
                if balls[j].isBall(ab[i]):
                    ab[i].id = j+1;
                    ab[i].healthy = balls[j].healthy
            if ab[i].id == None:
                ab[i].id = len(balls) + 1
                if nored and ab[i].isBall(red):
                    ab[i].healthy = False
                    nored = False
                balls.append(ab[i])
    was = True
    while not nored and was:
        was = False
        for i in range(len(ab)-1):
            for j in range(i+1, len(ab), 1):
                b1 = ab[i]
                b2 = ab[j]
                if (not b1.healthy and b2.healthy) or (b1.healthy and not b2.healthy):
                    if distance([b1.x, b1.y], [b2.x, b2.y]) < b1.r + b2.r + 10:
                        ab[i].healthy = False
                        ab[j].healthy = False
                        balls[b1.id - 1].healthy = False
                        balls[b2.id - 1].healthy = False
                        was = True
    encircle(img, ab)

    cv2.putText(img, str(len(balls)), (10, 30), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 0, 150), 1, cv2.LINE_AA)

    dim = resize(img, 0.5)
    cv2.imshow("img", dim)
    out.write(img)
    cv2.waitKey(1)

my_file = open("result.txt", "w")
my_file.write(str(len(balls)))
my_file.write("\n")
if nored:
    my_file.write("0")
else:
    my_file.write("1")
my_file.write("\n")
for b in balls:
    if not b.healthy:
        my_file.write(str(b.id))
        my_file.write("; ")
my_file.close()


cap.release()
out.release()
cv2.destroyAllWindows()

