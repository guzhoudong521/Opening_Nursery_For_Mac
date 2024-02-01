import signal
import time
from multiprocessing import Process, Queue, active_children
import pyautogui
import cv2
import numpy as np
import pytesseract
import Quartz


APP_NAME = "小程序"
APP_SHOT_FILENAME = "screenshot.png"
ORIGIN_WIDTH = 441
OFFSET_X = 12
OFFSET_TOP = 126
OFFSET_BOTTOM = 33
GRID_SIZE = 32
GRID_GAP = 10

def _getMousePosByGridPos(appInfo, gridPos, needOffset=False):
    (appX, appY, scale) = appInfo
    y, x = gridPos
    scaledOffsetX = OFFSET_X * scale
    scaledOffsetTop = OFFSET_TOP * scale
    scaledGridSize = GRID_SIZE * scale
    scaledGridGap = GRID_GAP * scale
    mouseX = appX + (
        scaledOffsetX + x * scaledGridSize + scaledGridGap * x + scaledGridGap
    )
    mouseY = appY + (
        scaledOffsetTop + y * scaledGridSize + scaledGridGap * y + scaledGridGap
    )

    if needOffset:
        mouseX += int(scaledGridSize / 2)
        mouseY += int(scaledGridSize / 2)

    return [mouseX, mouseY]

def _findRectangle(chessboard, taskQueue, row, col, startY, startX, endY, endX):
    for i in range(startY, endY + 1):
        for j in range(startX, endX + 1):
            for k in range(i, endY + 1):
                for l in range(j, endX + 1):
                    if i <= k and j <= l:
                        rectangle_sum = sum(
                            chessboard[x][y] for x in range(i, k + 1) for y in range(j, l + 1)
                        )
                        if rectangle_sum == 10:
                            taskQueue.put(([i, j], [k, l]), False)
                            for x in range(i, k + 1):
                                for y in range(j, l + 1):
                                    chessboard[x][y] = 0

def _queueTask(chessboard, taskQueue):
    row = len(chessboard)
    col = len(chessboard[0])
    for startY in range(row):
        for startX in range(col):
            _findRectangle(chessboard, taskQueue, row, col, startY, startX, row - 1, col - 1)

def _processTask(appInfo, taskQueue):
    guiStarted = False
    while True:
        try:
            task = taskQueue.get(block=False)
            guiStarted = True
            fromCell, toCell = task
            fromPos = _getMousePosByGridPos(appInfo, fromCell)
            toPos = _getMousePosByGridPos(appInfo, toCell, True)
            print("从", fromCell, "到", toCell)
            # print("拖拽 %s 到 %s" % (fromPos, toPos))
            pyautogui.moveTo(fromPos)
            time.sleep(0.06)
            pyautogui.dragTo(toPos, button='left', duration=0.3)
        except Exception as e:
            print(e)
            if guiStarted:
                break

def _stopProcess(signal, frame):
    print("Caught Ctrl+C, stopping processes...")
    for p in active_children():
        p.terminate()
    exit(0)

def appShot():
    all_apps = Quartz.NSWorkspace.sharedWorkspace().runningApplications()
    target_app = None
    for app in all_apps:
        if app.localizedName() == APP_NAME:
            target_app = app
            break
    if not target_app:
        print("ERROR: NO APP FOUND")
        exit()

    options = Quartz.kCGWindowListOptionOnScreenOnly | Quartz.kCGWindowListExcludeDesktopElements
    window_list = Quartz.CGWindowListCopyWindowInfo(options, target_app.processIdentifier())

    # 通过PID查找窗口 由于微信有多个进程，需要找到PID仅重复一次的进程 即为小程序
    all_pid = []
    for window in window_list:
        if window.get('kCGWindowOwnerName', '') != '微信':
            continue
        window_pid = window.get('kCGWindowOwnerPID', '')
        all_pid.append(window_pid)

    target_pid = None
    # 查找PID仅重复一次的进程
    for pid in all_pid:
        if all_pid.count(pid) == 1:
            target_pid = pid
            break
    if not target_pid:
        print("找不到微信窗口，请确保微信已打开小程序")
        exit()


    for window in window_list:
        window_pid = window.get('kCGWindowOwnerPID', '')
        if window_pid == target_pid:  # Modify the window PID accordingly
            window_frame = window.get('kCGWindowBounds', None)
            if window_frame:
                left = int(window_frame['X'])
                top = int(window_frame['Y'])
                width = int(window_frame['Width'])
                height = int(window_frame['Height'])

                # Click to activate the window
                pyautogui.click(left + 10, top + 10)

                im = pyautogui.screenshot(region=(left, top, width, height))
                im.save(APP_SHOT_FILENAME)

                scale = width / ORIGIN_WIDTH

                return (left, top, scale)

    print("找不到微信窗口，请确保微信已打开小程序")
    exit()

def ocr():
    img = cv2.imread(APP_SHOT_FILENAME)
    height, width, _ = img.shape

    scale = width / ORIGIN_WIDTH

    img = img[
        int(OFFSET_TOP * scale) : int(height - OFFSET_BOTTOM * scale),
        int(OFFSET_X * scale) : int(width - OFFSET_X * scale),
    ]

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    thresh = cv2.bitwise_not(thresh)
    # cv2.imwrite("debug.png", thresh)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    numbers = []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cell_min_size = GRID_SIZE * 0.9 * scale
        cell_max_size = GRID_SIZE * 1.1 * scale
        if w > cell_min_size and h > cell_min_size and w < cell_max_size and h < cell_max_size:
            number = img[y : y + h, x : x + w]
            if number.size > 0:
                x, y, w, h = 0, 0, number.shape[1], number.shape[0]
                offset = int(((GRID_SIZE - 20) / 2) * scale)
                number = number[y + offset : y + h - offset, x + offset : x + w - offset]
                number = cv2.resize(
                    number,
                    (
                        int(GRID_SIZE / 1.8 * scale),
                        int(GRID_SIZE / 1.8 * scale),
                    ),
                )
                numbers.append(number)
    if len(numbers) > 0:
        numbers = cv2.hconcat(numbers)
    else:
        raise Exception("识别失败")
    numbers = pytesseract.image_to_string(
        numbers, config='--psm 7 digits -c tessedit_char_whitelist="123456789"'
    )
    try:
        numbers = np.array([int(i) for i in list(numbers.strip())])
        numbers = np.flip(numbers)
        matrix = np.reshape(numbers, (16, 10))
    except:
        print("识别失败")
        return np.array([])
    return matrix

def auto(appInfo, matrix):
    chessboard = matrix
    print(chessboard)
    taskQueue = Queue()
    proc = []
    queueTask = Process(target=_queueTask, args=(chessboard, taskQueue))
    queueTask.start()
    proc.append(queueTask)
    processTask = Process(
        target=_processTask,
        args=(
            appInfo,
            taskQueue,
        ),
    )
    processTask.start()
    proc.append(processTask)
    signal.signal(signal.SIGINT, _stopProcess)
    for p in proc:
        p.join()

def start():
    appInfo = appShot()
    chessboard = ocr()
    if chessboard.size > 0:
        auto(appInfo, chessboard)
    else:
        print("没有识别到棋盘")

if __name__ == "__main__":
    start()
