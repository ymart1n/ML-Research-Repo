from YoloDetect import YoloDetect

def main():

    pic_path = 'sheep.jpg'
    td = YoloDetect(pic_path)

    result = td.TargetResult()
    print(result)

if __name__ == '__main__':
    main()